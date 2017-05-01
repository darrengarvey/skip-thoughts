#!/usr/bin/env python

"""
Convert a csv file into tf.Example records. Optionally preprocesses and cleans
text columns, if specified.
"""

import langdetect
import nltk.tokenize
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from six.moves.cPickle import dump
from tensorflow.python.platform import tf_logging as logging
from unidecode import unidecode

logging.set_verbosity(logging.DEBUG)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean("clean", True, "Clean text of text columns")
tf.app.flags.DEFINE_boolean("to_sequence_example", False,
                            "Convert to tf.SequenceExample instead of tf.Example")
tf.app.flags.DEFINE_boolean("remove_empty", False,
                            "Remove all rows that have an empty values for any "
                            "property in `text_columns`.")
tf.app.flags.DEFINE_boolean("remove_newlines", True,
                            "Remove all newlines (and duplicate whitespace) from "
                            "within `text_columns`.")
tf.app.flags.DEFINE_boolean("shuffle", True, "Shuffle examples")
tf.app.flags.DEFINE_boolean("lowercase", False, "Lower case text")
tf.app.flags.DEFINE_boolean("add_delimiters", True,
                            "Add end-of-sentence and end-of-document markers")
tf.app.flags.DEFINE_boolean("split_sentences", False, "Split input into sentences")
tf.app.flags.DEFINE_boolean("split_words", True, "Split input into word tokens.")
tf.app.flags.DEFINE_integer("num", -1, "Number of examples to output, -1=all")
tf.app.flags.DEFINE_float("train_frac", 0.9, "Fraction of examples for training. "
                          "Rest are evaluation")
tf.app.flags.DEFINE_string("input_file", "", "Input file to convert into tf.Examples")
tf.app.flags.DEFINE_string("column_output", None, "Write this column out to a text file")
tf.app.flags.DEFINE_string("train_output", "train.tfexample", "Training review examples")
tf.app.flags.DEFINE_string("eval_output", "eval.tfexample", "Evaluation review examples")
tf.app.flags.DEFINE_string("int_columns", "", "Comma-separated list of columns to treat"
                           "as integers.")
tf.app.flags.DEFINE_string("text_columns", "", "Comma-separated list of columns to treat"
                           "as text and clean, split, encode, etc.")
tf.app.flags.DEFINE_string("language", "",
                           "Allowed languages (eg. en). Remove other reviews. This is slow!")
tf.app.flags.DEFINE_string("vocab_path", None,
                           "If set, generate a vocab file and encode reviews using it.")


def remove_leading_and_trailing_whitespace(df, col):
    df[col] = df[col].str.strip()
    return df

def remove_duplicate_whitespace(df, col):
    df[col].replace(regex=True, inplace=True, to_replace=r'[\n\r ]+', value=' ')
    return df

def remove_empty_rows(df, col):
    original_len = len(df)
    df = df[df[col].str.len() != 0]
    logging.info('Found %d reviews that are empty',
                 original_len - len(df))
    return df

def remove_duplicate_sentence_ends(df, col):
    df[col].replace(regex=True, inplace=True, to_replace=r'[.!]+', value='.')
    return df

def remove_duplicate_question_marks(df, col):
    df[col].replace(regex=True, inplace=True, to_replace=r'[?]+', value='?')
    return df

def remove_non_ascii_chars(df, col):
    df[col] = df[col].apply(unidecode)
    return df

def lowercase_col(df, col):
    df[col] = df[col].str.lower()
    return df

def remove_non_ascii_rows(df, col):
    original_len = len(df)
    df = df[~df[col].str.contains("[\x7F-\xFFF]").fillna(False)]
    logging.info('Found %d reviews containing non-ascii characters',
                 original_len - len(df))
    return df

def remove_foreign_words(df, language, col):
    original_len = len(df)
    def lang_filter(review, lang):
        try:
            return langdetect.detect(review) == lang
        except:
            logging.error('Error %s', review)
            return False
    df = df[df.apply(lambda x: lang_filter(x[col], language), axis=1)]
    logging.info('Found %d reviews containing non-English words',
                 original_len - len(df))
    return df


def clean(df, cols):
    for col in cols:
        if not col in df.columns:
            logging.warn('Column not found: %s', col)
            continue
        df = remove_leading_and_trailing_whitespace(df, col)
        if FLAGS.remove_empty:
            df = remove_empty_rows(df, col)
        df = remove_non_ascii_chars(df, col)
        df = remove_duplicate_sentence_ends(df, col)
        df = remove_duplicate_question_marks(df, col)
        if FLAGS.lowercase:
            df = lowercase_col(df, col)
        if FLAGS.remove_newlines:
            df = remove_duplicate_whitespace(df, col)
        if FLAGS.language != '':
            df = remove_foreign_words(df, FLAGS.language, col)
    return df


def split_sentences(df, col):
    if not col in ['content']:
        return df
    sentences = df[col].apply(lambda s: nltk.tokenize.sent_tokenize(s))\
                       .apply(pd.Series, 1).stack()
    sentences.index = sentences.index.droplevel(-1)
    del(df[col])
    sentences.name = col
    df = df.join(sentences)
    return df


# These tokens won't be split by nltk.word_tokenize, so if you want to change
# them, remember to keep this property.
EOS_MARKER = ' _EOS_ '
EOF_MARKER = ' _EOF_ '

def write_examples(df, text_cols, output_name):
    columns = list(df.columns)
    if FLAGS.column_output:
        col = FLAGS.column_output
        with open(output_name, 'w') as f:
            for line in df[col]:
                for line in nltk.tokenize.sent_tokenize(line):
                    if col in text_cols and FLAGS.split_words:
                        line = ' '.join(nltk.tokenize.word_tokenize(line))
                    f.write(line + '\n')
        return

    int_cols = FLAGS.int_columns.split(',')
    with tf.python_io.TFRecordWriter(output_name) as writer:
        for _, row in df.iterrows():
            if FLAGS.to_sequence_example:
                ex = tf.train.SequenceExample()
                for col in columns:
                    if col in text_cols:
                        for sentence in nltk.tokenize.sent_tokenize(row[col]):
                            f = ex.feature_lists.feature_list[col].feature.add()
                            for word in nltk.tokenize.word_tokenize(sentence):
                                f.bytes_list.value.append(word)
                            if FLAGS.add_delimiters:
                                f.bytes_list.value.append(EOS_MARKER)
                        if FLAGS.add_delimiters:
                            f.bytes_list.value.append(EOF_MARKER)
                    else:
                        ex.context.feature[col].bytes_list.value.append(str(row[col]))
                writer.write(ex.SerializeToString())
            else:
                ex = tf.train.Example()
                for col in columns:
                    if FLAGS.vocab_path and col.endswith("_id") and col.strip("_id") in text_cols:
                        # The words are encoded, so store them as such.
                        for word_id in row[col]:
                            ex.features.feature[col].int64_list.value.append(word_id)
                        continue
                    elif col in int_cols:
                        ex.features.feature[col].int64_list.value.append(int(row[col]))
                        continue

                    text = str(row[col])
                    if FLAGS.add_delimiters and col in text_cols:
                        if FLAGS.split_sentences:
                            text += EOS_MARKER
                            # Don't put end of document markers in this case as we don't know
                            # when the sentences end anymore.
                        else:
                            text = EOS_MARKER.join(nltk.tokenize.sent_tokenize(text))
                            text += EOF_MARKER
                    if FLAGS.split_words and col.strip("_id") in text_cols:
                        text = ' '.join(nltk.tokenize.word_tokenize(text))
                        ex.features.feature[col].bytes_list.value.append(text)
                    else:
                        ex.features.feature[col].bytes_list.value.append(text)
                writer.write(ex.SerializeToString())


# Unused here. TF can do all this for us.
def text_to_id(df, column_name, train_output):
    values = df[column_name].str.strip().unique()
    value_to_id = dict(zip(values, range(1, len(values) + 1)))
    df[column_name + '_id'] = df[column_name].apply(lambda s: value_to_id[s.strip()])
    with open(train_output + '.' + column_name + '.id', 'w') as f:
        for k, v in value_to_id.iteritems():
            f.write('"%s",%d\n' % (k, v))
    return df


def encode_words(df, col, vocab_path):
    # The id in the vocab file of the end-of-sentence token
    EOS_ID = 0
    # The id in the vocab file of the unknown word token
    UNK_ID = 1
    word_to_id = {}
    sentences = df[col]
    if os.path.exists(vocab_path):
        logging.info('Using existing vocab file: %s', vocab_path)
        # Load the existing vocab file
        with open(vocab_path, 'r') as f:
            word_to_id.update({
                l.strip(): i
                for i, l in enumerate(f.readlines())
            })
    else:
        vocab_path='{}.{}'.format(vocab_path, col)
        logging.info('Generating vocab file: %s', vocab_path)
        # build word maps
        words = sentences.apply(lambda s: nltk.tokenize.word_tokenize(s))\
                         .apply(pd.Series, 1).stack().unique()
        word_to_id = dict(zip(words, range(1, len(words) + 1)))
        id_to_word = dict(zip(range(1, len(words) + 1), words))
        with open(vocab_path, 'wb') as f:
            dump([word_to_id, id_to_word], f)
        UNK_ID = 0 

    # split sentences into word lists and map to ids
    words = sentences.apply(lambda s: [word_to_id.get(w, 1)
                                       for w in nltk.tokenize.word_tokenize(s)])
    df["%s_id" % col] = words
    return df

def main(_):
    logging.info('Read file: %s', FLAGS.input_file)
    if FLAGS.input_file.endswith(".csv"):
        df = pd.read_csv(FLAGS.input_file)
    else:
        df = pd.read_json(FLAGS.input_file, lines=True)
    logging.info('Read %d rows' % len(df))
    logging.info('Cleaning')
    cols = FLAGS.text_columns.split(',')
    df = clean(df, cols)
    if FLAGS.split_sentences:
        for col in cols:
            logging.info('Split sentences in column: %s', col)
            df = split_sentences(df, col)
        logging.info('Now %d rows' % len(df))
    if FLAGS.shuffle:
        logging.info('Shuffling')
        df = df.iloc[np.random.permutation(len(df))]
    if FLAGS.num >= 0:
        logging.info('Sample %d rows' % FLAGS.num)
        df = df.sample(FLAGS.num)
    if FLAGS.vocab_path:
        for col in cols:
            df = encode_words(df, col, FLAGS.vocab_path)

    nrows = len(df)
    ntrain = int(float(FLAGS.train_frac) * nrows)
    rows = df.iloc[:ntrain]
    logging.info('Writing training examples (%d)' % len(rows))
    write_examples(rows, cols, FLAGS.train_output)
    rows = df.iloc[ntrain:]
    logging.info('Writing eval examples (%d)' % len(rows))
    write_examples(rows, cols, FLAGS.eval_output)
    logging.info('Done')


if __name__ == '__main__':
    tf.app.run()

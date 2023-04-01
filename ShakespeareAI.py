# %%
import numpy as np
import os
import tensorflow as tf

# %%
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# %%
text = open(path_to_file,'rb').read().decode(encoding ='utf-8')

# %%
vocab = sorted(set(text))

chars2ids = {unique:ids for ids, unique in enumerate(vocab)}
ids2chars = np.array(vocab)

text_as_int = np.array([chars2ids[char] for char in text])

# %%
seq_len = 100
examples_per_epoch = len(text) // (seq_len + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_len+1, drop_remainder = True)

# %%
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# %%
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# %%
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                    batch_input_shape = [batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences = True, stateful = True, 
                    recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(vocab_size) 
        ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = BATCH_SIZE)


# %%
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)


# %%
model.compile(optimizer='adam',loss=loss)
checkpoint_dir = '.\checkpoint_dir'
checkpoint_prefix = os.path.join(checkpoint_dir, 'chkpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_prefix,
        save_weights_only = True)
EPOCHS=50

history = model.fit(dataset,epochs=EPOCHS,callbacks=[checkpoint_callback])

# %%
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))


# %%
def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [chars2ids[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 0.25

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predictions = predictions // temperature
        predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(ids2chars[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string = 'ROMEO: '))



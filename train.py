import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
BATCH = 32
EXTRA_EPOCHS = 8
DATA_DIR = "dataset/UTKFace"
SEED = 42

H5_PATH = "age_gender_model.h5"
KERAS_PATH = "age_gender_model.keras"

def load_image_paths(directory):
    image_paths = []
    for fname in os.listdir(directory):
        low = fname.lower()
        if low.endswith((".jpg", ".jpeg", ".png")) or low.endswith(".jpg.chip.jpg"):
            image_paths.append(os.path.join(directory, fname))
    return image_paths

image_paths = load_image_paths(DATA_DIR)
if len(image_paths) == 0:
    raise ValueError(f"No images found in {DATA_DIR}")

tf.random.set_seed(SEED)
image_paths = tf.random.shuffle(
    tf.constant(image_paths), seed=SEED
).numpy().tolist()

val_ratio = 0.2
val_size = int(len(image_paths) * val_ratio)

val_paths = image_paths[:val_size]      # 20%
train_paths = image_paths[val_size:]    # 80%

def parse_age_gender_from_path(path):
    filename = tf.strings.split(path, os.sep)[-1]
    parts = tf.strings.split(filename, "_")
    age = tf.strings.to_number(parts[0], out_type=tf.int32)
    gender = tf.strings.to_number(parts[1], out_type=tf.int32)
    return age, gender

def load_img_and_labels(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    age, gender = parse_age_gender_from_path(path)
    return img, {"gender": gender, "age": tf.cast(age, tf.float32)}

def make_ds(paths, training=True):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        ds = ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_img_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(train_paths, training=True)
val_ds = make_ds(val_paths, training=False)

loaded = False
if os.path.exists(KERAS_PATH):
    model = load_model(KERAS_PATH, compile=False)
    loaded = True
elif os.path.exists(H5_PATH):
    model = load_model(H5_PATH, compile=False)
    model.save(KERAS_PATH)
    loaded = True
else:
    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_aug(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    gender_out = layers.Dense(2, activation="softmax", name="gender")(x)
    age_out = layers.Dense(1, activation="linear", name="age")(x)

    model = models.Model(inputs=inputs, outputs={
        "gender": gender_out,
        "age": age_out
    })

lr = 1e-5 if loaded else 1e-3
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss={
        "gender": "sparse_categorical_crossentropy",
        "age": "mse"
    },
    metrics={
        "gender": ["accuracy"],
        "age": ["mae"]
    }
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EXTRA_EPOCHS
)

model.save(KERAS_PATH)
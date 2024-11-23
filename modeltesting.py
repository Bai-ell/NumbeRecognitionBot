# import os
# import io
# import numpy as np
# from tensorflow.keras.models import load_model, Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.datasets import mnist
# from PIL import Image

# # Загрузка модели (если модель уже обучена, иначе можно использовать код ниже для обучения модели)
# try:
#     model = load_model("model2.h5")
#     print("Модель загружена успешно.")
# except:
#     print("Модель не найдена, обучаем новую.")

#     # Используем CNN для улучшения точности распознавания
#     model = Sequential([
#         Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         Conv2D(64, (3, 3), activation='relu'),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(10, activation='softmax')
#     ])

#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])

#     # Подготовка данных для обучения
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0  # Нормализация

#     # Преобразуем в формат (N, 28, 28, 1) для CNN
#     x_train = np.expand_dims(x_train, axis=-1)
#     x_test = np.expand_dims(x_test, axis=-1)

#     # Создаём генератор данных с аугментацией
#     datagen = ImageDataGenerator(
#         rotation_range=10,  # Повороты
#         width_shift_range=0.1,  # Сдвиг по горизонтали
#         height_shift_range=0.1,  # Сдвиг по вертикали
#         zoom_range=0.1,  # Масштабирование
#         shear_range=0.1,  # Наклон
#         horizontal_flip=False,  # Нет переворота
#         fill_mode='nearest'  # Заполнение пустых пикселей
#     )

#     # Применяем аугментацию к обучающим данным
#     datagen.fit(x_train)

#     # Обучаем модель с аугментацией
#     model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=5)  # Используем flow для аугментации данных

#     model.save("model2.h5")
#     print("Модель обучена и сохранена.")

# # Функция для обработки изображения
# def preprocess_image(photo_bytes):
#     img = Image.open(io.BytesIO(photo_bytes)).convert('L')  # Преобразуем изображение в оттенки серого
#     img = img.resize((28, 28))  # Изменяем размер на 28x28 (формат для модели)
#     img = img.point(lambda p: p > 128 and 255)  # Преобразуем в черно-белое изображение (бинаризация)
#     img_array = np.array(img) / 255.0  # Нормализуем изображение
#     img_array = np.expand_dims(img_array, axis=0)  # Добавляем батч
#     img_array = np.expand_dims(img_array, axis=-1)  # Добавляем канал для модели CNN
#     return img_array

# # Функция для предсказания
# def predict_digit(photo_bytes):
#     img_array = preprocess_image(photo_bytes)
#     predictions = model.predict(img_array)
#     predicted_digit = np.argmax(predictions)  # Получение цифры с максимальной вероятностью
#     return predicted_digit

# # Чтение изображения и предсказание
# def process_image(file_path):
#     with open(file_path, 'rb') as f:
#         photo_bytes = f.read()  # Чтение изображения в байтах
#     digit = predict_digit(photo_bytes)
#     print(f"Распознанная цифра на изображении {file_path}: {digit}")

# # Путь к папке с изображениями
# folder_path = "/Users/apple/python/myDevs/TemplatesForDevs/NUmberRecognitions/fotos"

# # Получаем все изображения в папке
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.jpg', '.JPG', '.png', '.jpeg')):  # Фильтруем только изображения
#         file_path = os.path.join(folder_path, filename)
#         process_image(file_path)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Загрузка и нормализация данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Построение модели
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Сохранение модели
model.save('model.h5')

# Оценка модели
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность на тестовых данных: {test_acc}")

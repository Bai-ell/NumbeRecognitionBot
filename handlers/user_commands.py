import io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from aiogram import Router, F
from aiogram.types import Message, FSInputFile, BufferedInputFile
from aiogram.types import InputFile
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.exceptions import TelegramAPIError
from PIL import Image
import logging

# Инициализация логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
try:
    model = load_model("model.h5")
    logger.info("Модель успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    raise

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Инициализация Router
router = Router()

# Функция для предсказания цифры по индексу
async def predict_from_index(n: int):
    try:
        if n < 0 or n >= len(x_test):
            raise ValueError("Индекс вне диапазона")
        
        # Получаем изображение и подготавливаем его для модели
        img_array = np.expand_dims(x_test[n], axis=0)  # Подготовка изображения
        res = model.predict(img_array)
        predicted_digit = np.argmax(res)
        
        return predicted_digit, x_test[n]
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise

# Функция для предсказания цифры на изображении
async def predict_from_image(image_data: bytes):
    try:
        # Преобразуем байты изображения в объект Image
        img = Image.open(io.BytesIO(image_data)).convert('L')  # Конвертируем в оттенки серого
        img = img.resize((28, 28))  # Меняем размер до 28x28 пикселей (размер MNIST изображений)
        
        # Преобразуем изображение в массив numpy и нормализуем
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
        
        # Прогнозируем цифру
        res = model.predict(img_array)
        predicted_digit = np.argmax(res)
        
        return predicted_digit
    except Exception as e:
        logger.error(f"Ошибка при предсказании с изображения: {e}")
        raise

# Обработчик команды /start
@router.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()
    await message.reply('Добро пожаловать! \nОтправь мне число (индекс от 0 до 999), чтобы выбрать изображение для распознавания, или отправь изображение с цифрой.')

# Обработчик для текста (индекса)
@router.message(F.text)
async def handle_number(message: Message):
    try:
        # Получаем число (индекс) от пользователя
        n = int(message.text)
        
        # Получаем предсказание и изображение для индекса
        predicted_digit, image = await predict_from_index(n)

        # Отправляем текст с предсказанием
        await message.reply(f"Распознанная цифра: {predicted_digit}")

        # Создаем изображение с помощью matplotlib
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.binary)  # Используем черно-белую палитру
        ax.axis('off')  # Убираем оси

        # Сохраняем изображение в файл
        img_path = 'predicted_image.png'
        plt.savefig(img_path, format='png')
        plt.close(fig)  # Закрываем график

        file = FSInputFile(img_path)  # Отправляем изображение
        await message.answer_photo(photo=file, caption="Вот ваше изображение!")

    except ValueError:
        await message.reply("Пожалуйста, отправьте число.")
    except Exception as e:
        await message.reply(f"Произошла ошибка: {e}")

# Обработчик для изображения (цифры)
# Обработчик для изображений, отправленных как фотографии
@router.message(F.photo)
async def handle_photo(message: Message):
    try:
        # Получаем самое большое изображение из списка
        photo = message.photo[-1]  # Самое большое изображение в списке
        
        # Получаем объект файла с помощью метода get_file()
        file_info = await message.bot.get_file(photo.file_id)
        
        # Загружаем изображение
        file = await message.bot.download_file(file_info.file_path)

        # Преобразуем изображение в байты
        image_data = file.getvalue()

        # Получаем предсказание для изображения
        predicted_digit = await predict_from_image(image_data)

        # Отправляем предсказанную цифру
        await message.reply(f"Распознанная цифра на изображении: {predicted_digit}")

    except Exception as e:
        await message.reply(f"Произошла ошибка при обработке фотографии: {e}")

# Обработчик для файлов (например, jpeg, jpg)
@router.message(F.document)
async def handle_document(message: Message):
    try:
        # Проверка, что файл является изображением
        file_name = message.document.file_name.lower()
        if file_name.endswith(('jpeg', 'jpg', 'png')):
            # Получаем объект файла с помощью метода get_file()
            file_info = await message.bot.get_file(message.document.file_id)
            
            # Загружаем файл
            file = await message.bot.download_file(file_info.file_path)

            # Преобразуем изображение в байты
            image_data = file.getvalue()

            # Получаем предсказание для изображения
            predicted_digit = await predict_from_image(image_data)

            # Отправляем предсказанную цифру
            await message.reply(f"Распознанная цифра на изображении: {predicted_digit}")

        else:
            await message.reply("Пожалуйста, отправьте изображение в формате JPEG, PNG или JPG.")

    except Exception as e:
        await message.reply(f"Произошла ошибка при обработке файла: {e}")


# Обработчик любых других сообщений
@router.message()
async def handle_other_messages(message: Message):
    await message.reply("Пожалуйста, отправьте число от 0 до 999 или изображение с цифрой для распознавания.")

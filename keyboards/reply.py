from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardRemove
)




async def main_keyboard(): 
    
    main = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text='Проверить цифру')
               
            ],   
        ],
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder= 'Выюерите действие',
    )
    return main

async def main():
    keyboard = await main_keyboard()
    
    
rmk = ReplyKeyboardRemove()

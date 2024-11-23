import asyncio
from aiogram import Bot, Dispatcher
import logging
from config_reader import config
from handlers import user_commands, callback_handlers
from middlewares.antiflood import AntiFloodMiddleware
from middlewares.badwordshandler import MultiLangBadWordsMiddleware


#example main
async def main() -> None:
    bot = Bot(config.bot_token.get_secret_value())
    dp = Dispatcher()

    #example middlewares 

    #example badwords json files
    # dp.message.middleware(MultiLangBadWordsMiddleware(file_paths=["badwordsru.json"]))
    #example Delay for bot in seconds
    dp.message.middleware(AntiFloodMiddleware(0.5))

    #example files router
    dp.include_routers(
        user_commands.router,
        callback_handlers.router,
     
    )

    # example webhook settings
    await bot.delete_webhook(drop_pending_updates=True)

    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Exiting')
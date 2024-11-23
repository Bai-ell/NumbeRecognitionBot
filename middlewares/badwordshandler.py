from typing import Callable, Awaitable, Dict, Any, List
from aiogram import BaseMiddleware
from aiogram.types import Message
import json



#example Multi languages badwords handler if you add file in main.py string 19
class MultiLangBadWordsMiddleware(BaseMiddleware):
    def __init__(self, file_paths: List[str]) -> None:
        self.bad_words = []
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    self.bad_words.extend(data.get("bad_words", []))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    
    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:
       
        if isinstance(event, Message):
            message_text = event.text 

            
            if any(bad_word in message_text for bad_word in self.bad_words):
                await event.answer('Dont say that')
                return

        return await handler(event, data)
import os

class Storage:
    def __init__(self, id: str):
        self.id = id
        # Create chats directory if it doesn't exist
        if not os.path.exists("chats"):
            os.makedirs("chats")
    
    def save_chat_history(self, chats: list, language: str = "en"):
        """
        Save chat history for specific language
        
        Args:
            chats: List of chat messages
            language: Language code (e.g., 'gaa', 'tw', 'en')
        """
        try:
            # Create language-specific file
            filename = f"chats/{self.id}_{language}.txt"
            
            for chat in chats:
                with open(filename, "a", encoding='utf-8') as file:
                    # Try to get translated content first, fallback to original
                    lang_key = f"content_{language}"
                    content = chat.get(lang_key, chat.get('content', ''))
                    
                    # Write to file
                    role = chat['role'].capitalize()
                    file.write(f"{role}: {content}\n")
                    
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def load_chat_history(self, language: str = "en") -> list:
        """
        Load chat history for specific language
        
        Args:
            language: Language code
            
        Returns:
            List of chat messages
        """
        try:
            filename = f"chats/{self.id}_{language}.txt"
            
            if not os.path.exists(filename):
                return []
            
            messages = []
            with open(filename, "r", encoding='utf-8') as file:
                lines = file.readlines()
                
                for line in lines:
                    if line.strip():
                        # Parse the line
                        if ": " in line:
                            role, content = line.split(": ", 1)
                            role = role.lower()
                            if role == "user" or role == "ai" or role == "assistant":
                                if role == "assistant":
                                    role = "ai"
                                messages.append({
                                    "role": role,
                                    "content": content.strip(),
                                    f"content_{language}": content.strip()
                                })
            
            return messages
            
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return []
    
    def clear_chat_history(self, language: str = None):
        """
        Clear chat history for specific language or all languages
        
        Args:
            language: Language code or None to clear all
        """
        try:
            if language:
                # Clear specific language file
                filename = f"chats/{self.id}_{language}.txt"
                if os.path.exists(filename):
                    os.remove(filename)
            else:
                # Clear all language files for this chat ID
                import glob
                pattern = f"chats/{self.id}_*.txt"
                files = glob.glob(pattern)
                for file in files:
                    os.remove(file)
                    
        except Exception as e:
            print(f"Error clearing chat history: {e}")
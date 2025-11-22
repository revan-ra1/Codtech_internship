import random
import string
from datetime import datetime

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# -------------------------------------------------
# NLTK downloads (run once; then cached)
# -------------------------------------------------
def ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


ensure_nltk_data()

# -------------------------------------------------
# NLP helpers
# -------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(token):
    """Map POS tag to first character used by WordNetLemmatizer."""
    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess(text: str):
    """
    Lowercase, remove punctuation, tokenize, remove stopwords,
    and lemmatize tokens. Returns a set of important words.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)

    filtered = []
    for w in tokens:
        if w in stop_words:
            continue
        pos = get_wordnet_pos(w)
        lemma = lemmatizer.lemmatize(w, pos=pos)
        filtered.append(lemma)

    return set(filtered)


# -------------------------------------------------
# Chatbot definition
# -------------------------------------------------

class Chatbot:
    def __init__(self):
        # Intents: general conversation
        self.intents = {
            "greeting": {
                "patterns": [
                    "hi",
                    "hello",
                    "hey",
                    "good morning",
                    "good evening",
                    "yo",
                ],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hi there 😀 What would you like to know?",
                    "Hey! Ask me anything.",
                ],
            },
            "goodbye": {
                "patterns": [
                    "bye",
                    "goodbye",
                    "see you",
                    "see ya",
                    "exit",
                    "quit",
                ],
                "responses": [
                    "Goodbye! Have a great day 😊",
                    "See you soon!",
                    "Bye! It was nice chatting with you.",
                ],
            },
            "name": {
                "patterns": [
                    "what is your name",
                    "who are you",
                    "tell me about yourself",
                ],
                "responses": [
                    "I'm a simple NLP chatbot built using Python and NLTK.",
                    "You can call me RevBot 🤖.",
                ],
            },
            "help": {
                "patterns": [
                    "help",
                    "what can you do",
                    "how can you help me",
                    "what are your features",
                ],
                "responses": [
                    "I can answer basic questions, greet you, tell you about myself, "
                    "give you time/date, and respond to some simple FAQs.",
                    "Try asking: 'what is your name?', 'what is NLP?', or just say 'hi'.",
                ],
            },
            "thanks": {
                "patterns": [
                    "thank you",
                    "thanks",
                    "thx",
                ],
                "responses": [
                    "You're welcome! 😊",
                    "No problem at all!",
                    "Any time!",
                ],
            },
            "mood": {
                "patterns": [
                    "how are you",
                    "how are you doing",
                    "how is it going",
                ],
                "responses": [
                    "I'm just code, but I'm running perfectly! How about you?",
                    "I'm doing great inside this terminal. 😄 How are you?",
                ],
            },
        }

        # FAQ / knowledge base (very simple)
        self.faq_pairs = {
            "what is nlp": "NLP (Natural Language Processing) is a field of AI that helps computers understand human language.",
            "what is python": "Python is a high-level, interpreted programming language known for readability and ease of use.",
            "what is machine learning": "Machine learning is a subset of AI where systems learn patterns from data instead of being explicitly programmed.",
            "what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
            "who created you": "I was created as a demo chatbot using Python and NLTK.",
        }

        # Preprocess intent patterns
        self._intent_patterns_processed = []
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data["patterns"]:
                self._intent_patterns_processed.append(
                    {
                        "intent": intent_name,
                        "pattern": pattern,
                        "tokens": preprocess(pattern),
                    }
                )

        # Preprocess FAQ keys
        self._faq_processed = []
        for question, answer in self.faq_pairs.items():
            self._faq_processed.append(
                {
                    "question": question,
                    "answer": answer,
                    "tokens": preprocess(question),
                }
            )

    # ---------------- Intent matching ---------------- #

    def detect_time_or_date(self, user_input: str):
        text = user_input.lower()
        if "time" in text:
            now = datetime.now().strftime("%H:%M:%S")
            return f"The current time is {now}."
        if "date" in text or "day" in text:
            today = datetime.now().strftime("%Y-%m-%d")
            return f"Today's date is {today}."
        return None

    def match_intent(self, user_input: str, threshold: float = 0.2):
        input_tokens = preprocess(user_input)
        if not input_tokens:
            return None

        best_intent = None
        best_score = 0.0

        for item in self._intent_patterns_processed:
            overlap = input_tokens.intersection(item["tokens"])
            if not item["tokens"]:
                continue
            score = len(overlap) / len(item["tokens"])
            if score > best_score:
                best_score = score
                best_intent = item["intent"]

        if best_score >= threshold:
            return best_intent
        else:
            return None

    def match_faq(self, user_input: str, threshold: float = 0.3):
        input_tokens = preprocess(user_input)
        if not input_tokens:
            return None

        best_answer = None
        best_score = 0.0

        for item in self._faq_processed:
            overlap = input_tokens.intersection(item["tokens"])
            if not item["tokens"]:
                continue
            score = len(overlap) / len(item["tokens"])
            if score > best_score:
                best_score = score
                best_answer = item["answer"]

        if best_score >= threshold:
            return best_answer
        else:
            return None

    # ---------------- Response generation ---------------- #

    def generate_response(self, user_input: str) -> str:
        # 1. time/date special case
        td_answer = self.detect_time_or_date(user_input)
        if td_answer:
            return td_answer

        # 2. Try intent
        intent = self.match_intent(user_input)
        if intent:
            responses = self.intents[intent]["responses"]
            return random.choice(responses)

        # 3. Try FAQ
        faq_answer = self.match_faq(user_input)
        if faq_answer:
            return faq_answer

        # 4. Fallback
        return (
            "I'm not sure I understood that 🤔.\n"
            "You can ask me about AI, NLP, Python, or type 'help' to see what I can do."
        )

    # ---------------- Chat loop ---------------- #

    def chat(self):
        print("==========================================")
        print("🤖  RevBot – AI Chatbot with NLP (NLTK)")
        print("Type 'bye' or 'exit' to quit.")
        print("Type 'help' to see what I can do.")
        print("==========================================\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["bye", "exit", "quit"]:
                goodbye_responses = self.intents["goodbye"]["responses"]
                print("Bot:", random.choice(goodbye_responses))
                break

            bot_reply = self.generate_response(user_input)
            print("Bot:", bot_reply)


if __name__ == "__main__":
    bot = Chatbot()
    bot.chat()

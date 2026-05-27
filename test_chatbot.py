from ml.chatbot import smart_chat

print("Asking about budget...")
res1 = smart_chat("how do you know if money is being misused on a road?")
print("Bot:", res1)

print("\nAsking a generic question...")
res2 = smart_chat("can you explain more about that?")
print("Bot:", res2)

print("\nAsking about pothole...")
res3 = smart_chat("I saw a huge pothole on my street today")
print("Bot:", res3)

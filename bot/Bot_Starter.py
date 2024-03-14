import discord
import asyncio 
from openai import OpenAI
import sys
sys.path.append('../model')
from naive_bayes import Bayes_Classifier


gptClient = OpenAI(api_key="API_KEY_HERE")

bot_token = 'DISCORD_TOKEN_HERE'
intents = discord.Intents.all()
intents.message_content == True
client = discord.Client(command_prefix='!', intents=intents)

model = Bayes_Classifier(train_dir = "../model/data/train")
model.loadModel()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    message.content = message.content.lower()
    model = Bayes_Classifier(train_dir = "../model/data/train")
    model.loadModel()
    guh = model.classify(message.content)
    if message.author == client.user:
        return
    elif 'Gorp/reload' in message.content:
        model.loadModel()
    elif 'whiny' == model.classify(message.content) :
        response = gptClient.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a sassy discord bot, you shut down whiny messages by sassing them and then telling them to shut up without saying shut up. Stay on topic with their whiny message."},
                {"role": "system", "content": "Respond in one, one line message with no quotes and make sure to shut them down"},
                {"role": "system", "content": "put an emoji between every word but don't repeat emojis and make sure to be on topic to what the message was"},
                {"role": "system", "content": "swear in all your responses but instead of actual swear words use an euphemism"},
                {"role": "user", "content": f"{message.content}"}
                ]
        )

        await message.channel.send(response.choices[0].message.content)
client.run(bot_token)
#!/usr/bin/env python
# coding: utf-8

# # Script zum automatisierten Abzug von Kursdaten & Tweet-Klassifizierung zu Kryptowährung

# Das folgende Script soll auf einem Linuxserver via Cronjob automatisiert alle 30 Minuten ausgeführt werden.  
# Im ersten Schritt werden bei jeder Ausführung die aktuellen Kursdaten (bspw. aktueller Kurs, Marktkapitalisierung) zu den relevantesten Krypto Währungen von der Internetseite https://crypto.com/price abgezogen, aufbereitet und in eine .csv Datei abgespeichert.  
# Anschließend werden über die Twitter-API Tweets zu den zuvor ausgelesenen Kryptowährungen abgezogen und mit Hilfe der Bibliothek "TextBlob" einer Sentiment Analysis unterzogen.
#   
# Bei jeder Ausführung des Scripts werden die neuen Informationen (Kursdaten & Tweet-Klassifizierungen) in die jeweiligen .csv Dateien angehängt, sodass eine fortlaufende Historie entsteht.  
#   
# Die hierbei entstehenden .csv Dateien sollen dann im Anschluss zur Analyse verwendet werden. Die Analyse findet in einem anderen Jupyter Notebook statt, welches nicht auf dem Server ausgeführt werden muss.

# ## 1. Setup

# Zuerst müssen die notwendigen Module oder libraries importiert werden. der folgende Code ist nur für die Ausführung auf einem lokalen Rechner relevant. Damit das Script auf dem Linux Server läuft, wurden die Module manuell via pip auf dem Server installiert.

# In[1]:


#Basics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv

#Webcrawling
#pip install beautifulsoup4
from bs4 import BeautifulSoup
import requests


# In[2]:


# Für Twitter Abzug & Klassifizierung
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from textblob import TextBlob 


# ## 2. Skript zum automatisierten Abzug der Währungsinformationen

# ### 2.1 Webcrawling

# Zu Beginn wird eine Funktion definiert, welche die aktuellen Kursdaten zu den relevantesten Kryptowährungen von der Internetseite https://crypto.com/price abzieht. 

# In[5]:


# Erstellen der Funktion zum crawlen der aktuellen Kursdaten
def get_crypto(URL):
    CryptoDF=pd.DataFrame()
    soup=BeautifulSoup(requests.get(URL).text,"html.parser")
    # in der folgenden Schleife werden aus den relevanten html Elementen "table" die benötigten Informationen 
    # zum zuvor erstellen "CryptoDF" hinzugefügt
    for currency in soup.find("table" , {"class":"chakra-table css-1qpk7f7"}).find_all("tr"):
        CryptoList = []
        for c in currency.findAll(["p","span","div","td"]):
            if c.text.startswith(".css"):
                continue
            else:
                CryptoList.append(c.text)
        try:
            CryptoDF = CryptoDF.append({"Pos":CryptoList[3], "Name":CryptoList[11], "Short":CryptoList[12], "Price in $":CryptoList[14], "24h Change in %":CryptoList[17], "24h Volume in M$":CryptoList[20], "Market Cap in M$":CryptoList[21]}, ignore_index=True)
        except:
            continue 
        # zuletzt sollen nur die relevantesten 10 Währungen übernommen werden
        # Relevanz wird hier als Marktkapitalisierung definiert, wonach die auf der Webseite aufgeführten Währungen bereits sortiert sind
        CryptoDF= CryptoDF.head(10)
    return CryptoDF


# Nun kann die Funktion mit dem entsprechenden Link als Attribut aufgerufen werden, um so einen aktuellen Abzug der Kursdaten zu erzeugen:

# In[6]:


CryptoDF = get_crypto("https://crypto.com/price")


# In[7]:


CryptoDF


# ### 2.2 Methode zum Erstellen der Historie

# Als Nächstes sollen die abgezogenen Kursdaten zu einer Historie hinzugefügt werden, welche so den Kursverlauf über längere Zeit widerspiegelt.

# Folgende Codezeile ließt aus einer der "HistoryDF.csv" Datei das initial erstellte Dataframe ein. An diese .csv Datei werden in den folgenden Schritten die gecrawlten Daten angehängt und wieder abgespeichert. So entesteht in der "HistoryDF.csv" eine fortlaufende Historie der Währungsdaten.

# In[8]:


HistoryDF= pd.read_csv("HistoryDF.csv", index_col=0)
HistoryDF


# Im nächsten Schritt wird eine Funktion definiert, welche zu einem im Funktionsaufruf anzugebenden Attribut aus dem obigen "CryptoDF" (bspw. "Price in $") eine Zeile mit Zeitstempel erstellt.  
# Diese Zeile kann dann an das HistoryDF angehängt werden.

# In[9]:


# Funktion zum Erstellen eines Abzuges eines beliebigen Wertes mit aktueller Systemzeit 
def create_snapshot(attribute):
    TransposedDF=CryptoDF[["Short",attribute]].transpose().reset_index().rename(columns={'index':'var'})
    TransposedDF.columns = TransposedDF.iloc[0]
    TransposedDF = TransposedDF.drop(TransposedDF.index[0])
    TransposedDF = TransposedDF.rename(columns={"Short": "ValueCategory"})
    TransposedDF["timestamp"] = datetime.now()
    return TransposedDF


# Hier wird die Funktion nur beispielhaft aufgerufen, in der späteren Anwendung findet zuvor eine Bereinigung des CryptoDF statt, weswegen das im HistoryDF zu sehende Format etwas von dem Format aus der folgenden Codezeile abweicht.

# In[10]:


# Beispiel Nutzung der create_snapshot Funktion zum Abzug des Wertes "Price"
create_snapshot("Price in $")


# ### 2.3 Anwenden der Methode, Erstellen der Historie

# Nun werden die zuvor definierten Funktionen angewendet.  
# Zuerst wird die Funktion zum crawlen der aktuellen Währungsinformationen ausgeführt.

# In[11]:


CryptoDF = get_crypto("https://crypto.com/price")
CryptoDF


# Störende Zeichen werden an dieser Stelle bereits entfernt:

# In[12]:


CryptoDF = CryptoDF.replace(['%','\$','\,'], '', regex=True)
CryptoDF


# Um die Spalten "24h Volume" und "Market Cap" in einen Zahlenwert umzuwandeln, müssen die Buchstaben "B" (Billion/Milliarden) und "M" (Million/Millionen) umgewandelt werden.
# Das wird in der folgenden Funktion umgesetzt. Zahlen mit "B" werden mit 1000 multipliziert und Zahlen mit M so wie sie sind wieder ins CryptoDF eingesetzt. So werden die Werte in Millionen $ dargestellt.

# In[13]:


# B/Milliarden wird mit 1000 multipliziert und M/Millionen mit 1, sodass der Wert am Ende in Millionen angegeben ist
def transformvalues(columnname):
    list = []
    for val in CryptoDF[columnname]:
        split = val.split(' ')
        parsed_value = float(split[0])
        if split[1] == 'B':
            list.append(int(parsed_value * 1000))
        elif split[1] == 'M':
            list.append(int(parsed_value))
        else:
                print('error')            
    CryptoDF[columnname] = list


# In[14]:


transformvalues("Market Cap in M$")
transformvalues("24h Volume in M$")


# In[15]:


CryptoDF


# Zum Schluss wird nun die zuvor definierte "create_snapshot" Funktion für die relevanten Spalten aufgerufen und die dadurch erzeugten Zeilen ans HistoryDF angefügt.  
# Danach wird das erweiterte HistoryDF wieder als .csv abgespeichert, sodass es im nächsten Durchgang wieder eingelesen und erweitert werden kann.

# In[16]:


HistoryDF = HistoryDF.append(create_snapshot("Price in $"),ignore_index=True)
HistoryDF = HistoryDF.append(create_snapshot("24h Volume in M$"),ignore_index=True)
HistoryDF = HistoryDF.append(create_snapshot("24h Change in %"),ignore_index=True)
HistoryDF = HistoryDF.append(create_snapshot("Market Cap in M$"),ignore_index=True)
HistoryDF.to_csv("HistoryDF.csv")
HistoryDF


# ## 3. Script zum automatisierten Abzug von Tweets

# ### 3.1 Methode zum Abrufen der Twitter API

# Zu Beginn wird die bereits erstellte CSV-Datei eingelesen. Diese wurde initial mit allen benötigten Spalten angelegt und hier eingelesen. In den folgenden Codezeilen wird der Inhalt erstellt, welcher am Ende des Notebooks wieder in die CSV-Datei geschrieben wird.

# In[285]:


TwitterDF= pd.read_csv("TwitterDF.csv", index_col=0)


# Nun wird das BearerToken festgelegt, welches benötigt wird, um die Twitter API v2 zu nutzen.  
# Hierfür wurde ein Account auf https://developer.twitter.com mit Essential Access angelegt.

# In[286]:


BearerToken = 'AAAAAAAAAAAAAAAAAAAAANUSkAEAAAAAPuizqhzPfOPmzsUISkQoxq1zi%2BI%3DuAlv2ZACzeHIINh7xzXyqDgVZDvonTsN667e5vsoohZNDlXOri'


# Für den Abzug von Tweets muss zunächst eine Client-Verbindung unter Verwendung des Bearer Token hergestellt werden.

# In[287]:


# create your client 
client = tweepy.Client(BearerToken)


# Nachfolgend wird eine Methode zum Crawlen von Tweets definiert. Dieser muss ein Suchkriterium (hier *crypto* ) mitgegeben werden, welches in den Tweets enthalten sein muss. Zudem ist eine Start- und Endzeit der Suche mitzugeben sowie die Anzahl maximaler Resultate. Der Essential Access erlaubt eine maximale Anzahl von 100 Tweets pro Request, eine Anzahl von 10 ist das Minimum.  
# In der Methode wird die Query durch das Suchkriterium, der Sprache Englisch und der Deaktivierung von Retweets erstellt. Diese sowie alle weiteren Parameter werden dann an die Methode *search_recent_tweets* des Clients übergeben. Dadurch crawlt die Methode Tweets, die den Suchkriterien entsprechen.  
# Anschließend wird über die Tweets iteriert und ihre Erstellzeit *created_at* sowie den Inhalt des Tweets *text* in ein Dictionary geschrieben. Diese wird um die Information des Suchkriteriums *crypto* erweitert.  
# Die Methode gibt das Dictionary als Rückgabewert zurück.

# In[288]:


def getTweets(crypto, start_time, end_time, max_results):
    query = crypto + ' lang:en -is:retweet'
    tweets = client.search_recent_tweets(query=query,
                                     start_time=start_time,
                                     end_time=end_time,
                                     tweet_fields = ["created_at", "text", "source"],
                                     max_results = max_results
                                    )      
    tweet_info_ls = []
    # iterate over each tweet and corresponding details
    for tweet in tweets.data:
        tweet_info = {
            'created_at': tweet.created_at,
            'text': tweet.text,
        }
        tweet_info['crypto'] = crypto
        tweet_info_ls.append(tweet_info)
    return tweet_info_ls


# Aus dem zuvor erstellten CryptoDF werden die Namen aller Kryptowährungen in einer Liste gespeichert.

# In[289]:


cryptoList= CryptoDF["Name"]
cryptoList


# ### 3.2 Ausführen der Methode & Abspeichern der Daten

# Zur Vorbereitung des Methodenaufrufs werden die Parameter initialisiert. Als Startzeit wird die aktuelle Zeit minus 30 Minuten gesetzt (=Zeitpunkt des letzten Abzugs), Endzeit ist der aktuelle Zeitstempel. Als maximale Ergebnisanzahl wird das Maximum des Essential Access Kontos zur Twitter API v2 gesetzt (100) und ein leeres Dataframe erzeugt.

# In[298]:


start_time = datetime.now() - timedelta(hours=0, minutes=30)
end_time = datetime.now()
max_results = 100
tweets_df = pd.DataFrame()


# Nun wird über die Liste alle Kryptowährungen iteriert und der jeweilige Name der Kryptowährung der Methode *getTweets* als Parameter mitgegeben. Somit ist der jeweilige Name der Kryptowährung der Suchbegriff im Abzug der Tweets. Das Ergebnis wird dem eben initialisierten Dataframe *tweets_df* angehängt.  
# In der nächsten Iteration geschieht das gleiche mit der nächsten Kryptowährung. Somit werden für jede Kryptowährung 100 Tweets abgezogen und im Dataframe tweets_df gespeichert.

# In[ ]:


for crypto in cryptoList:
    tweet_liste = getTweets(crypto, start_time, end_time, max_results=max_results)
    tweets_df = tweets_df.append(tweet_liste, ignore_index=True)


# Das Dataframe wird anschließend als .csv gespeichert.

# In[300]:


tweets_df.to_csv("Tweets.csv")


# In[3]:


tweets_df


# ### 3.3 Sentiment Analysis

# Ziel des nächsten Schritts ist es, die eben abgezogenen Tweets in *positiv* , *negativ* und *neutral* zu klassifizieren. Dafür wird die Python Bibliothek "TextBlob" genutzt. (https://textblob.readthedocs.io/en/dev/quickstart.html)  
# 
# Die Tweets, welche der Methode als Parameter zusammen mit dem Namen der jeweiligen Kryptowährung mitgegeben werden, werden zunächst in Textblob Objekte umgewandelt.  
# Für diese kann die Property .sentiment aufgerufen werden, welche ein Tupel aus Polarity und Subjectivity zurückgibt. Beides sind float Werte, wobei Polarity in der Range [-1.0 , 1.0] angegeben wird und kleine Werte negative Äußerungen und große Werte für positive stehen. Subjectivity wird in der Range [0.0 , 1.0] angegeben, wobei 0.0 sehr objektiv und 1.0 sehr subjektiv ist.  
# In diesem Projekt ist nur die Polarity von Interesse, weshalb nur dieser Wert pro Tweet in der Liste *sentiment_values* gespeichert wird.  
# Anschließend wird über die Liste iteriert und Werte größer Null als positiv, Werte kleiner Null als negativ und Werte gleich Null als neutral abgespeichert.  
# In einem Dictionary wird der aktuelle Zeitstempel, der Name der Kryptowährung sowie die Anzahl an positiver, negativer und neutraler Tweets gespeichert. Zudem wird der Wert *count* als die Anzahl aller untersuchten Tweets gespeichert.  
# Das Dictionary ist der Rückgabewert der Methode.

# In[303]:


def sentimentClassification(tweets, crypto):
    # Classify
    sentiment_objects = [TextBlob(tweet) for tweet in tweets]
    # Create a list of polarity values and tweet text
    sentiment_values = [tweet.sentiment.polarity for tweet in sentiment_objects]
    # Initialize variables, 'pos', 'neg', 'neu'.
    pos=0
    neg=0
    neu=0

    #Create a loop to classify the tweets as Positive, Negative, or Neutral.
    # Count the number of each.

    for items in sentiment_values:
        if items>0:
            pos=pos+1
        elif items<0:
            neg=neg+1
        else:
            neu=neu+1
        
    data = {
            'time': datetime.now(),
            'cypto': crypto,
            'pos': pos,
            'neg': neg,
            'neu': neu,
            'count': len(tweets)
        }

    return data


# Die eben definierte Methode muss nun je Kryptowährung aufgerufen werden. Dafür wird wieder über die cryptoList iteriert und das tweets_df auf die Tweets der jeweiligen Kryptowährung eingeschränkt. Diese Tweets werden der Methode sentimentClassification zusammen mit dem Namen der Kryptowährung als Parameter mitgegeben. Das zurückgegebene Dictionary wird dem TwitterDF angehängt.  

# In[304]:


for crypto in cryptoList:
    tweets = tweets_df['text'].loc[tweets_df['crypto'] == crypto]
    data = sentimentClassification(tweets, crypto)
    TwitterDF = TwitterDF.append(data, ignore_index=True)


# Das TwitterDF wird in der CSV-Datei gespeichert, welche zu Beginn des Twitter-Teils im Notebook eingelesen wurde.  
# Da das Notebook alle 30 Minuten ausgeführt wird, wird das TwitterDF somit alle 30 Minuten erweitert um eine Zeile je Kryptowährung mit aktuellem Zeitstempel und der Anzahl positiver, negativer und neutraler Tweets je Währung.  
# Diese Informationen können in einem lokalen Notebook analyisiert werden.

# In[305]:


TwitterDF.to_csv("TwitterDF.csv")


# In[306]:


TwitterDF


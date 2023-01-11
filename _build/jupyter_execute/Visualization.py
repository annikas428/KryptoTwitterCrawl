#!/usr/bin/env python
# coding: utf-8

# # Web & - Social Media Analytics Projekt:
# # Analyse von Kryptowährung Kursdaten & Tweets

# *von Annika Scheug und Oliver Schabe*

# Ziel des hier umgesetzten Projektes im Bereich Web- & Social Media Analytics ist die Gewinnung und Auswertung von Informationen zum Thema Kryptowährung.  
# Das hier erstellte Prototyp Tool soll mögliche Nutzer befähigen, aktuelle Informationen zum Marktverhalten der relevantesten Kryptowährungen abzurufen. Diese Informationen könnten beispielweise unterstützen bei der Entscheidung zum Kauf oder Verkauf von Tokens/Coins oder einfach nur einen Überblick über den Markt geben.  
# Als Datenquellen wird die Internetseite https://crypto.com/price nach relevanten Informationen gecrawlt sowie aktuelle Tweets zu den entsprechenden Kryptowährungen über die Twitter-API abgezogen. Die abgezogenen Tweets sollen daraufhin einer simplen Sentiment Analysis unterzogen werden, um einen Überblick über mehrheitlich positive oder negative Tweets zu den Währungen zu erhalten.  
# Die Daten werden kontinuierlich in regelmäßigen Abständen abgezogen und abgespeichert, wodurch eine Historie mit Preisverläufen etc. aufgebaut wird (Siehe 2. Jupyter Notebook).  
# Aus der Kombinationen dieser Daten sollen wenn möglich Erkenntnisse zu Abhängigkeiten zwischen Tweets und Kursinformationen gewonnen werden, wie beispielsweise viele positive Tweets und ein steigender Währungskurs.

# ## 1. Setup

# Zu Beginn werden die benötigten Libraries und Module importiert.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
# Für interaktive Plots
#pip install jupyter-dash
from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ## 2. Einlesen und Verknüpfung der Daten

# Im ersten Schritt werden die CSV-Dateien, welche auf dem Linuxserver alle 30 Minuten um neue Daten erweitert werden (s. 2. Jupyter NB), in dieses Notebook als Dataframe eingelesen.

# In[2]:


TwitterDF = pd.read_csv('TwitterDF.csv', index_col=0)
HistoryDF = pd.read_csv('HistoryDF.csv', index_col=0)


# In[4]:


HistoryDF


# In[3]:


TwitterDF


# Die Timestamps der Dataframes werden auf Minutengenauigkeit abgeschnitten.

# In[4]:


TwitterDF['time_short'] = TwitterDF.apply(lambda x: x['time'][:-10], axis = 1)
HistoryDF['time_short'] = HistoryDF.apply(lambda x: x['timestamp'][:-10], axis = 1)


# Mit der Methode .info() werden die Spalten eines Dataframe mit ihren Datentypen und evtl. Nullwerten angezeigt.

# In[5]:


TwitterDF.info()


# Im Dataframe gibt es keine Nullwerte. Die Timestamps sind als Datentyp object abgespeichert. Dies wird im Nachfolgenden noch auf Datentyp datetime geändert.

# In[6]:


TwitterDF['time_short'] = pd.to_datetime(TwitterDF['time_short'])
HistoryDF['time_short'] = pd.to_datetime(HistoryDF['time_short'])


# In[7]:


TwitterDF.describe()


# Im Durchschnitt liegen 43 positive, 16 negative und 40 neutrale Tweets pro Zeile des Dataframes vor.  
# Variable Count zählt die Anzahl der Tweets pro Zeile. Meistens wurden 100 Tweets abgezogen (Einstellung des max_results Werts im Twitterabzug auf dem Server). In einigen wenigen Fällen scheinen weniger als 100 Tweets den Suchkriterien entsprochen zu haben, weshalb nicht alle 100 Tweets gecrawlt wurden. Dies ist am Durchschnittswert von Count mit einem Wert von 99,21 zu erkennen. Um die Anzahl positiver und negativer Tweets zwischen allen Währungen und Zeitstempeln vergleichbar zu machen, wird daher der prozentuale Anteil dieser beiden Werte errechnet und dem Dataframe hinzugefügt.

# In[8]:


TwitterDF['percentage_pos'] = TwitterDF['pos']/TwitterDF['count']*100
TwitterDF['percentage_neg'] = TwitterDF['neg']/TwitterDF['count']*100


# Die Twitterdaten und Kryptowährungsdaten liegen aktuell in 2 unterschiedlichen Dataframes vor. Ziel ist die gemeinsame Analyse der Werte. Daher werden diese nun in ein Dataframe gejoined.

# Dafür wird zunächst eine Methode definiert, welche zu einer als Parameter mitgegebenen Kryptowährung ein eigenens Dataframe erzeugt.  
# Dieser wird der prozentuale Anteil positiver und negativer Tweets sowie der gekürzte Timestamp als jeweils eigene Spalte zugewiesen. Das erzeugte Dataframe mit 3 Spalten ist der Rückgabewert der Methode.

# In[9]:


def createCryptoDF(crypto):
    df = pd.DataFrame()
    cryptoDF = TwitterDF.loc[TwitterDF['crypto'] == crypto]
    df[crypto + ' % pos Tweets'] = cryptoDF['percentage_pos'].loc[cryptoDF['crypto'] == crypto]
    df[crypto + ' % neg Tweets'] = cryptoDF['percentage_neg'].loc[cryptoDF['crypto'] == crypto]
    df['time_short'] = TwitterDF['time_short']
    return df


# Diese Methode wird anschließend für jede Kryptowährung aus dem TwitterDF aufgerufen. Der einzelnen Dataframes als Rückgabewerte der Methode werden an das JoinedDF über den gekürzten Timestamp time_short gemerged.  
# Als Resultat besteht das Joined DF aus dem ehemaligen HistoryDF sowie einer zusätzlichen Spalte "% pos Tweets" und "% neg Tweets" je Kryptowährung.  
# Existieren im HistoryDF Timestamps, welche nicht im TwitterDF exisitieren, so werden die Spalten "% pos/neg Tweets" mit Nullwerten belegt.  
# Da im HistoryDF jeder Zeitstempel viermal existiert (einmal pro ValueCategory), werden die Tweet-Daten an jeweils 4 Zeilen des HistoryDFs gejoined. Dieses Verhalten ist so gewünscht, da die Twitterdaten nun mit jeder ValueCategory verglichen werden können.

# In[10]:


CurrencyNames = HistoryDF.columns[:10]
crypto = TwitterDF['crypto'].unique()


# In[11]:


JoinedDF = HistoryDF
for cr in crypto:
    JoinedDF = pd.merge(JoinedDF,  createCryptoDF(cr), on="time_short", how="left")


# In[13]:


JoinedDF


# In[12]:


JoinedDF.info()


# Es liegen nun einige Nullwerte in den Tweet-Spalten vor. Das TwitterDF enthält weniger Zeitstempel HistoryDF. Grund dafür ist, dass das Python Skript auf dem Linuxserver zu Beginn einige Male abgebrochen ist und keine Twitter Daten gespeichert wurden.  
# Ursache hierfür war, dass das tweets_df als .csv abgespeichert und direkt danach wieder eingelesen wurde. Das Einlesen ist mit folgender Fehlermeldung abgebrochen:  
# *ParserError: Error tokenizing data. C error: Buffer overflow caught - possible malformed input file.*  
# Diese Problematik wurde nach einiger Zeit aber identifiziert und behoben, indem die .csv nicht nochmal eingelesen wurde. Alternativ hätte man der Methode pd.read_csv() noch folgenden Parameter mitgeben können: "lineterminator='\n',".  
# In Konsequenz werden während der Analysen im Dashboard Nullwerte gedropped, wenn Währungs- mit Twitterdaten verglichen werden. Ein Vergleich der Werte ist an Timestamps in denen nur Kryptodaten vorliegen nicht möglicht.

# Aus Gründen der Einfachheit in den Analysen wird das JoinedDF in einzelne Dataframes nach ValueCategory unterteilt.

# In[14]:


PriceDF = JoinedDF.loc[JoinedDF['ValueCategory'] == 'Price in $']
VolumeDF = JoinedDF.loc[JoinedDF['ValueCategory'] == '24h Volume in M$']
ChangeDF = JoinedDF.loc[JoinedDF['ValueCategory'] == '24h Change in %']
MarketCapDF = JoinedDF.loc[JoinedDF['ValueCategory'] == 'Market Cap in M$']


# ## 3. Funktionen zur Dashboard Erstellung

# Vor der Datenanalysen werden Methoden zum Plotten der Daten definiert.

# ### 3.1 Interaktive Dashboards mit Plotly Dash

# Zum erstellen von interaktiven Plots wird die Library Plotly Dash (bzw. Jupyter Dash) verwendet. Diese bietet die Möglichkeit, innerhalb eines Jupyter Notebooks eine interaktive App zur Visualisierung und Analyse von Daten zu erstellen.
# Um den Code nicht immer wieder erneut aufführen und anpassen zu müssen, werden im Folgenden zwei Funktion zur Erstellung einer solchen App definiert.  
# Für die zwei unterschiedlichen Datenquellen "Währungsdaten" und "Tweets" wird jeweils eine eigene Funktion definiert.  
# Eine Einschränkung besteht bei Plotly Dash darin, dass in Jupyter Notebook jederzeit nur eine Dash Applikation aktiv sein kann. Dazu später bei Anwendung der Funktion mehr.

# In[16]:


# Funktion zur interkativen Visualisierung von Währungsdaten
def build_dashboard(df,header):
    app = JupyterDash(__name__)
    #hier wird das Layout der App sowie Elemente des Plots wie Graph und Checkliste inkl. Parameter definiert
    app.layout = html.Div([
        html.H4(header),
        dcc.Graph(id="graph"),
        dcc.Checklist(
            id="checklist",
            options=CurrencyNames,
            value=["BTC"],
            inline=True
        ),
    ])
    # hier wird ein sogenannter callback definiert, eine Funktion die jedes mal aufgerufen wird wenn sich ein 
    # input Parameter in der App (in diesem Fall in der checklist) ändert und somit den Plot aktualisiert
    @app.callback(
        Output("graph", "figure"), 
        Input("checklist", "value"))
    def update_line_chart(selected_currency):
        fig = px.line(df, x="time_short", y=selected_currency)
        fig.update_layout(transition_duration=500)
        return fig
    return app


# In[24]:


# Funktion zur interkativen Visualisierung von Tweets
def build_dashboard_tweets(df,header, tweets):
    app = JupyterDash(__name__)

    app.layout = html.Div([
        html.H4(header),
        dcc.Graph(id="graph"),
        dcc.Checklist(
            id="checklist",
            options=tweets,
            #value=tweets,
            inline=True
        ),
    ])
    @app.callback(
        Output("graph", "figure"), 
        Input("checklist", "value"))
    def update_line_chart(selected_currency):
        fig = px.line(df, x="time_short", y=selected_currency)
        fig.update_layout(transition_duration=500)
        return fig
    return app


# ### 3.2 Plots mit multiplen Achsen

# Zusätzlich zu den interaktiven Plots werden hier zwei weitere Funktionen definiert, um Plots mit zwei verschiedenen Y-Achsen zu generieren.  
# Aufgrund großer Werteunterschiede in einzelnen Variablen werden verschiedene Skalierungen der Y-Achse in einem Plot benötigt, um Datenkurven miteinander zu vergleichen.  

# Zunächst wird die Methode merge_plots definiert, um zwei unterschiedliche ValueCategories einer Kryptowährung miteinander zu vergleichen.

# In[18]:


def merge_plots(df1,df2,currency):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df1["time_short"], y=df1[currency],name=df1.iloc[0].ValueCategory),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df2["time_short"], y=df2[currency],name=df2.iloc[0].ValueCategory),
        secondary_y=True,
    )
    # Add figure title


    # Set x-axis title
    fig.update_xaxes(title_text="Zeit")
    return fig


# Außerdem wird die gleiche Methode benötigt, um zwei unterschiedliche Spalten des gleichen Dataframes (also innerhalb der gleichen ValueCategory) zu vergleichen.

# In[19]:


def merge_plots2(df, column1,column2):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df["time_short"], y=df[column1],name=column1),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df["time_short"], y=df[column2],name=column2),
        secondary_y=True,
    )
    # Add figure title


    # Set x-axis title
    fig.update_xaxes(title_text="Zeit")
    return fig


# ## 1. Analyse der Tweets

# Die Spaltennamen der positiven und negativen Tweets werden in den Listen PosTweets und NegTweets gespeichert, um diese Spalten schnell referenzieren zu können.

# In[20]:


PosTweets = list(ChangeDF.filter(regex=("pos Tweets*")).columns)
NegTweets = list(ChangeDF.filter(regex=("neg Tweets*")).columns)
NegTweets


# Da nun alle Vorbereitungen getroffen sind, kann mit de explorativen Datenanalyse der Tweets begonnen werden.  

# ### 1.1 Verlauf positive Tweets

# Im ersten Plot sollen die positiven Tweets der Währungen 

# In[73]:


# WICHTIG: Es kann immer nur eine dash application gleichzeitig im Jupyter Notebook laufen.
# Wenn eine andere app gestartet wurde, wird in diesem Fenster ebenfalls die zuletzt gestartete app aufgeführt 
# (nach Aktualisierung)
# In diesem Fall einfach diese Code Zeile erneut ausführen
app1 = build_dashboard_tweets(PriceDF.dropna(),"% positive Tweets", PosTweets)
# Falls Fehlermeldung "Address 'http://localhost:XXXX' already in use" muss ein anderer Port angegeben werden
app1.run_server(debug=True, use_reloader=False,mode="inline", port="8851") 


# Im Plot können die positiven Tweets der einzelenen Kryptowährungen durch die Filter zu- und weggeschaltet werden.  
# Die positiven Tweets von Bitcoin liegen stets im Mittelfeld.  
# BNB sticht zu Beginn der Datenaufnahme durch viele positive Tweets heraus, ab dem 8. Januar schneidet diese Währung mittelmäßig ab. Stattdessen sticht ab diesem Zeitpunkt Dogegoin durch viele positive Tweets heraus.  
# Über Binance USD wurde über den kompletten Zeitraum wenig positives getweetet.

# ### 1.2 Verlauf negative Tweets

# Die gleiche Darstellung wird nun auf die negativen Tweets angewendet.

# In[78]:


# WICHTIG: Es kann immer nur eine dash application gleichzeitig im Jupyter Notebook laufen.
# Wenn eine andere app gestartet wurde, wird in diesem Fenster ebenfalls die zuletzt gestartete app aufgeführt 
# (nach Aktualisierung)
# In diesem Fall einfach diese Code Zeile erneut ausführen
app1 = build_dashboard_tweets(PriceDF.dropna(),"% negative Tweets", NegTweets)
# Falls Fehlermeldung "Address 'http://localhost:XXXX' already in use" muss ein anderer Port angegeben werden
app1.run_server(debug=True, use_reloader=False,mode="inline", port="8851") 


# Vom 5.-9. Januar zeigen sich bei Tether interessante Peeks der negativen Tweets zur Mittagszeit.  
# Über Bitcoin wird wenig negatives gepostet.  
# Besonders auffällig ist in diesem Plot wieder Binance USD, zu welcher Währung über den kompletten Zeitraum sehr viele negative Tweets gecrawlt wurden.  
# Zudem sticht Polygon heraus, da über diese Währung konstant wenig negatives getweetet wird, es jedoch nur am 8. Januar zu sehr vielen negativen Tweeets kam. Dies könnte auf extrem negative Nachrichten über die Kryptowährung am 8. Januar hindeuten.
# 
# 
# genauere Untersuchung von Bitcoin (als Referenzwährung), Binance USD, da eher negativ auffällig, BNB da eher positiv auffallend und Polygon wegen Ereignis am 8.1.23

# ### 1.3 Gegenüberstellung von Tweets ausgewählter Währungen

# In den vorherigen Plots sind einige Währungen durch Auffälligkeiten herausgestochen.  
# Dazu gehören Binance USD durch konstant negative Tweets, Polygon durch den extremen Peek negativer Tweets am 8.1. sowie BNB durch sehr viele positive Tweets.  
# Daher wird auf diese Währungen im weiteren Verlauf dieses Notebook besonders eingegangen. Zudem wird Bitcoin als Referenzwährung detaillierter untersucht.  

# Für die ausgewählten Währungen werden nachfolgend nochmal positive und negative Tweets im zeitlichen Verlauf gegenübergestellt.

# In[33]:


fig = px.line(ChangeDF.dropna(), x="time_short", y=['Bitcoin % pos Tweets', 'Bitcoin % neg Tweets'],width=865, height=450)
fig.show()


# Über Bitcoin wird konstant mehr positiv als negativ berichtet. Es scheint jedoch auch einige neutrale Tweets zu geben, da die Summe beider Prozentzahlen weniger als 100% ergibt.

# In[35]:


fig = px.line(ChangeDF.dropna(), x="time_short", y=['BNB % pos Tweets', 'BNB % neg Tweets'],width=865, height=450)
fig.show()


# Bei BNB ist die Differenz zwischen positiver und negativer Tweets größer als bei Bitcoin. 

# In[36]:


fig = px.line(ChangeDF.dropna(), x="time_short", y=['Binance USD % pos Tweets', 'Binance USD % neg Tweets'],width=865, height=450)
fig.show()


# Bincance USD scheint als Kryptowährung weniger beliebt zu sein, da verstärkt negativ berichtet wird.

# In[37]:


fig = px.line(ChangeDF.dropna(), x="time_short", y=['Polygon % pos Tweets', 'Polygon % neg Tweets'],width=865, height=450)
fig.show()


# Polygon sticht durch den 8. Januar heraus. Bis auf dieses Datum wurden deutlich mehr positive als negative Tweets veröffentlicht.

# ## 2. Übersicht Währungskurs

# ### 2.1 Aktueller Währungskurs

# Zunächst wird der aktuelle Währungskurs (zum Zeitpunkt des letzten Abzugs) in Dollar als Barchart visualisiert. So fallen besonders schnell die großen Unterschiede im Preis pro Token / Coin auf.

# In[106]:


fig = px.bar(PriceDF.iloc[0], x=CurrencyNames, y=PriceDF.iloc[0,0:10].values,text_auto=True)
fig.show()


# ### 2.2 Verlauf Währungskurs

# Zusätzlich zum aktuellen Währungskurs ist auch der Verlauf über einen längeren Zeitraum interessant, um zu erkennen wie sich Kryptowährungen entwickelt haben.  
# Da die Höhe des aktuellen Währungskurs zwischen den verschiedenen Währung extrem unterschiedlich ist (bspw Bitcoin 5-stellig vs. USDT 1-stellig) können im folgenden Plot die Währung über Filter ein- und ausgeschaltet werden. Die y-Achse passt sich dynamisch dem entsprechenden Werteintervall an.

# In[41]:


# WICHTIG: Es kann immer nur eine dash application gleichzeitig im Jupyter Notebook laufen.
# Wenn eine andere app gestartet wurde, wird in diesem Fenster ebenfalls die zuletzt gestartete app aufgeführt 
# (nach Aktualisierung)
# In diesem Fall einfach diese Code Zeile erneut ausführen
app1 = build_dashboard(PriceDF,"Verlauf Währungskurs in $")
# Falls Fehlermeldung "Address 'http://localhost:XXXX' already in use" muss ein anderer Port angegeben werden
app1.run_server(debug=True, use_reloader=False,mode="inline", port="8851") 


# Bitcoin geht stetig hoch
# BNB und ETH Kurve sehen ähnlich zu Bitcoin aus, nur value viel niedriger, aber ähnlicher verlauf
# USDT, USDC, BUSD und DOGE Kurs unverändert => Währungen orientieren sich an US-Dollar, daher keine großen Schwankungen
# Polygon (MATIC) geht am 9.1. hoch, kein negativer Kurs am 8.1. zu erkennen, welcher negative Tweets rechtfertigen würde, Cardano (ADA) Kurve sieht ähnlich aus

# ### 2.3 Vergleich Währungskurs mit anderen

# In[40]:


fig = merge_plots(PriceDF,ChangeDF,"BTC")
fig.update_yaxes(title_text="Währungskurs in $", secondary_y=False)
fig.update_yaxes(title_text="Wachstum in %", secondary_y=True)
fig.update_layout(title_text="Vergleich Währungskurs & Wachstum Bitcoin")
fig.show()


# In[42]:


fig = merge_plots(PriceDF,VolumeDF,"BNB")
fig.update_yaxes(title_text="Währungskurs in $", secondary_y=False)
fig.update_yaxes(title_text="24h Volume in M$", secondary_y=True)
fig.update_layout(title_text="Vergleich Währungskurs & 24h Volume BNB")
fig.show()


# ### 2.4 Vergleich Preisentwicklung mit Tweets

# In diesem Teil wird die Preisentwicklung mit den Tweets verglichen. Da sich im Plot mit multiplen Y-Achsen immer nur eine Währung darstellen lässt, werden in diesem Notebook nicht alle Kryptowährungen im Detail aufgeführt, sondern pro Value Category 2 der auffälligen Währung ausgewählt.  
# Dies ist durch Austausch der Übergabeparameter an die Methode merge_plots2 jederzeit anpassbar.

# In[43]:


fig = merge_plots2(PriceDF.dropna(),'BTC',"Bitcoin % pos Tweets")
fig.update_yaxes(title_text="Währungskurs in $", secondary_y=False)
fig.update_yaxes(title_text="% pos Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich Währungspreis & positive Tweets")
fig.show()


# Im Plot von Währungspreis und positiver Tweets und Bitcoin ist keine wirkliche Abhängigkeiten dieser beiden Werte zu erkennen. Die positiven Tweets von Bitcoin schwanken konstant in einem ähnlichen Spektrum.

# In[45]:


fig = merge_plots2(PriceDF.dropna(),'BUSD',"Binance USD % neg Tweets")
fig.update_yaxes(title_text="Währungskurs in $", secondary_y=False)
fig.update_yaxes(title_text="% pos Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich Währungspreis & negative Tweets")
fig.show()


# Der Preis von BUSD ist fast über den gesamten Zeitraum konstant. Daher ist keine Korrelation zur schwankenden Anzahl negativer Tweets erkennbar.  
# Allerdings könnte die fehlende positive Kursentwicklung auch ein Grund für die hohe Anzahl an negativen Tweets sein.

# ## 3. Übersicht Marktkapitalisierung

# ### 3.1 Aktuelle Marktkapitalisierung

# In[47]:


# Um die aktuelle Marktkapitalisiserung anzuzeigen, wird der neuste Werte (Index 0 = steht an erster Stelle) 
# aus dem DF zur Visualisierung ausgewählt
fig = px.pie(MarketCapDF.iloc[0], values = MarketCapDF.iloc[0,:10].values, names=CurrencyNames, title='Marktkapitalisierung in M$')
fig.show()


# ### 3.2 Verlauf Marktkapitalisierung

# In[48]:


# WICHTIG: Es kann immer nur eine dash application gleichzeitig im Jupyter Notebook laufen.
# Wenn eine andere app gestartet wurde, wird in diesem Fenster ebenfalls die zuletzt gestartete app aufgeführt 
# (nach Aktualisierung)
# In diesem Fall einfach diese Code Zeile erneut ausführen
app1 = build_dashboard(MarketCapDF,"Verlauf Marktkapitalisierung in M$")
# Falls Fehlermeldung "Address 'http://localhost:XXXX' already in use" muss ein anderer Port angegeben werden
app1.run_server(debug=True, use_reloader=False,mode="inline", port="8851") 


# ### 3.3 Vergleich Marktkapitalisierung mit anderen Informationen

# In[49]:


fig = merge_plots(MarketCapDF,PriceDF,"BTC")
fig.update_yaxes(title_text="Marktkapitalisierung in M$", secondary_y=False)
fig.update_yaxes(title_text="Währungskurs in $", secondary_y=True)
fig.update_layout(title_text="Vergleich Marktkapitalisierung & Wachstum Bitcoin")
fig.show()


# Währungskurs und Marktkapitalisierungskurven fast identisch (auch wenn Y-Achsen unterschiedlich), gleicher Verlauf => starke Korrelation erkennbar

# In[50]:


fig = merge_plots(MarketCapDF,PriceDF,"MATIC")
fig.update_yaxes(title_text="Marktkapitalisierung in M$", secondary_y=False)
fig.update_yaxes(title_text="Währungskurs in $", secondary_y=True)
fig.update_layout(title_text="Vergleich Marktkapitalisierung & Wachstum Polygon")
fig.show()


# Korrelation auch mit anderer Währung erkennbar

# In[118]:


fig = merge_plots(MarketCapDF,ChangeDF,"BTC")
fig.update_yaxes(title_text="Marktkapitalisierung in M$", secondary_y=False)
fig.update_yaxes(title_text="Wachstum in %", secondary_y=True)
fig.update_layout(title_text="Vergleich Marktkapitalisierung & Wachstum Bitcoin")
fig.show()


# Kurven sehen auch sehr ähnlich aus

# ### 3.4 Vergleich Marktkapitalisierung mit Tweets

# In[52]:


fig = merge_plots2(MarketCapDF.dropna(),'BNB',"BNB % pos Tweets")
fig.update_yaxes(title_text="Market Cap in M$", secondary_y=False)
fig.update_yaxes(title_text="% pos Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich Market Cap & positive Tweets")
fig.show()


# Die Anzahl positiver Tweets nimmt Ende Januar ab, dennoch steigt Marktkapitalisierung. Daher ist auch hier keine Korrelation zwischen positiver Tweets und Marktkapitalisierung erkennbar.   

# In[54]:


fig = merge_plots2(MarketCapDF.dropna(),'MATIC',"Polygon % neg Tweets")
fig.update_yaxes(title_text="Market Cap in M$", secondary_y=False)
fig.update_yaxes(title_text="% pos Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich Market Cap & negative Tweets")
fig.show()


# Der Peek von negativen Tweets am 8. Januar scheint nicht durch die Marktkapitalisierung erklärbar zu sein. Hier gab es an diesem Tag keine negativen Auffälligkeiten. Ab dem 9. Januar ist die Marktkapitalisierung sogar erheblich gestiegen, trotz vieler negativer Tweets am Vortag.

# ## 4. Übersicht Wachstum

# ### 4.1 Verlauf Wachstum

# In[55]:


# WICHTIG: Es kann immer nur eine dash application gleichzeitig im Jupyter Notebook laufen.
# Wenn eine andere app gestartet wurde, wird in diesem Fenster ebenfalls die zuletzt gestartete app aufgeführt 
# (nach Aktualisierung)
# In diesem Fall einfach diese Code Zeile erneut ausführen
app1 = build_dashboard(ChangeDF,"Verlauf Wachstum in %")
# Falls Fehlermeldung "Address 'http://localhost:XXXX' already in use" muss ein anderer Port angegeben werden
app1.run_server(debug=True, use_reloader=False,mode="inline", port="8851") 


# ### 4.2 Vergleich Wachstum mit anderen Informationen

# In[128]:


fig = merge_plots(ChangeDF,MarketCapDF,"MATIC")
fig.update_yaxes(title_text="Wachstum in %", secondary_y=False)
fig.update_yaxes(title_text="Marktkapitalisierung in M$", secondary_y=True)
fig.update_layout(title_text="Vergleich Wachstum & Marktkapitalisierung Polygon")
fig.show()


# In[129]:


fig = merge_plots(ChangeDF,VolumeDF,"MATIC")
fig.update_yaxes(title_text="Wachstum in %", secondary_y=False)
fig.update_yaxes(title_text="24h Volume in M$", secondary_y=True)
fig.update_layout(title_text="Vergleich Wachstum & 24h Volume Polygon")
fig.show()


# ### 4.3 Vergleich Wachstum mit Tweets

# In[58]:


fig = merge_plots2(ChangeDF.dropna(),'BUSD',"Binance USD % neg Tweets")
fig.update_yaxes(title_text="Wachstum in %", secondary_y=False)
fig.update_yaxes(title_text="% neg Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich Wachstum & negative Tweets")
fig.show()


# In diesem Plot lässt sich schon eher ein Zusammenhang erkennen. An einigen Zeitpunkten steigt das Wachstum und die negative Anzahl an Tweets sinkt. Ab dem 9. Januar ist das Wachstum in den negativen Bereich gefallen und die Anzahl negativer Tweets deutlich gestiegen.

# In[59]:


fig = merge_plots2(ChangeDF.dropna(),'MATIC',"Polygon % neg Tweets")
fig.update_yaxes(title_text="Wachstum in %", secondary_y=False)
fig.update_yaxes(title_text="% neg Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich Wachstum & negative Tweets")
fig.show()


# In diesem Plot ist zu erkennen, dass am 7. Januar eine Wachstumsphase begonnen hat, die dann am 8. Januar wieder etwas nachließ. Möglicherweise hat dies auch in das Verstärkte Auftreten der negativen Tweets am 8. Januar geführt. Ab dem 9. Januar ist das Wachstum wieder gestiegen und die Anzahl negativer Tweeets hat sich wieder auf einen niedrigen Wert eingependelt. Das erneute Einbrechen der Wachstumsrate am 10. Januar hat zu keiner Veränderung der Tweets geführt.

# ## 5. Übersicht 24h Volume

# ### 5.1 Verlauf 24h Volume

# In[64]:


# WICHTIG: Es kann immer nur eine dash application gleichzeitig im Jupyter Notebook laufen.
# Wenn eine andere app gestartet wurde, wird in diesem Fenster ebenfalls die zuletzt gestartete app aufgeführt 
# (nach Aktualisierung)
# In diesem Fall einfach diese Code Zeile erneut ausführen
app1 = build_dashboard(VolumeDF,"Verlauf 24h Volume in M$")
# Falls Fehlermeldung "Address 'http://localhost:XXXX' already in use" muss ein anderer Port angegeben werden
app1.run_server(debug=True, use_reloader=False,mode="inline", port="8851") 


# ### 5.2 Vergleich 24h Volume mit anderen Informationen

# In[135]:


fig = merge_plots(VolumeDF,ChangeDF,"USDT")
fig.update_yaxes(title_text="24h Volume in M$", secondary_y=False)
fig.update_yaxes(title_text="Wachstum in %", secondary_y=True)
fig.update_layout(title_text="24h Volume & Wachstum Tether")
fig.show()


# In[136]:


fig = merge_plots(VolumeDF,ChangeDF,"BNB")
fig.update_yaxes(title_text="24h Volume in M$", secondary_y=False)
fig.update_yaxes(title_text="Wachstum in %", secondary_y=True)
fig.update_layout(title_text="24h Volume & Wachstum BNB")
fig.show()


# Kurven wieder ähnlich

# ### Vergleich 24h Volume mit Tweets

# In[67]:


fig = merge_plots2(VolumeDF.dropna(),'ETH',"Ethereum % pos Tweets")
fig.update_yaxes(title_text="24h Volume in M$", secondary_y=False)
fig.update_yaxes(title_text="% pos Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich 24h Volume & positive Tweets")
fig.show()


# Zwischen 24h Volume und positiver Tweets von Ethereum ist kein Zusammenhang erkennbar.

# In[68]:


fig = merge_plots2(VolumeDF.dropna(),'BNB',"BNB % pos Tweets")
fig.update_yaxes(title_text="24h Volume in M$", secondary_y=False)
fig.update_yaxes(title_text="% pos Tweets", secondary_y=True)
fig.update_layout(title_text="Vergleich 24h Volume & positive Tweets")
fig.show()


# Auch bei BNB ist kein Zusammenhang zwischen 24h Volume und Anzahl positiver Tweets erkennbar.

# ## 6. Berechnung von Korrelationen

# In[141]:


# Create correlation matrix for numerical variables
corr_matrix = PriceDF[['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'XRP', 'BUSD', 'DOGE', 'ADA', 'MATIC', 'Bitcoin % pos Tweets', 'Bitcoin % neg Tweets']].corr()
corr_matrix['BTC'].sort_values(ascending=False)


# Wie schon in Plots festgestellt, sind zeigen viele Kryptowährungen ähnliche Verläufe im Währungspreis wie Bitcoin. Dies zeigt sich hier auch an der Korrelation. USDT, USDC und BUSD die sich am US-Dollar orientieren zeigen wenig bis kaum Korrelation
# Zur Anzahl positiver oder negativer Tweets von Bitcoin ist keine Korrelation zum Bitcoin Preis aus der Korrelationsmatrix erkennbar

# In[142]:


# Create correlation matrix for numerical variables
corr_matrix = MarketCapDF[['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'XRP', 'BUSD', 'DOGE', 'ADA', 'MATIC', 'BNB % pos Tweets', 'BNB % neg Tweets']].corr()
corr_matrix['BNB'].sort_values(ascending=False)


# Korrelation der Währungen untereinander erkennbar, negative Korrelation zu BNB % pos Tweets ergibt wenig Sinn, würde bedeuten je größer die Marktkapitalisierung desto weniger positive Tweets

# In[143]:


# Create correlation matrix for numerical variables
corr_matrix = VolumeDF[['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'XRP', 'BUSD', 'DOGE', 'ADA', 'MATIC', 'Binance USD % pos Tweets', 'Binance USD % neg Tweets']].corr()
corr_matrix['BUSD'].sort_values(ascending=False)


# Korrelation der Währungen untereinander, positive Korrelation zu Binance USD % negativer und positiver Tweets, auch wenig sinnvoll

# In[144]:


# Create correlation matrix for numerical variables
corr_matrix = ChangeDF[['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'XRP', 'BUSD', 'DOGE', 'ADA', 'MATIC', 'Polygon % pos Tweets', 'Polygon % neg Tweets']].corr()
corr_matrix['MATIC'].sort_values(ascending=False)


# wieder starke Korrelation der Währungen untereinander
# diesmal Korrelation ChangeDF zu Polygon % neg Tweets auch ganz gut erkennbar, negative Korrelation = positiver Change führt zu weniger negativer Tweets ==> erscheint logisch

# ## 7. Öffnen von Chrome zum Kauf von Kryptowährung mit Selenium

# In[47]:


lastChanges = ChangeDF[CurrencyNames].tail(1)
lastChanges


# In[52]:


cryptoBuy = lastChanges.idxmax(axis="columns")


# In[53]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys

#from selenium.webdriver.support.ui import Select

driver = webdriver.Chrome()
vars = {}
  
driver.get("https://www.bybit.com/fiat/trade/express/home")

driver.fullscreen_window()

box = driver.find_element(By.XPATH, '/html/body/div[3]/div[2]/div/section[2]/div/div/div/div[1]/div[1]/div[1]/div[2]/div/div/span[1]/input')
box.send_keys('100000')


driver.find_element(By.XPATH, '/html/body/div[3]/div[2]/div/section[2]/div/div/div/div[1]/div[2]/div/div[2]/div/div/span[2]/div/div/div/div/div/span[1]').click()
box2 = driver.find_element(By.XPATH, '/html/body/div[5]/div/div/div/div/div/div/span[2]/input')

box2.send_keys(cryptoBuy)


# In[153]:


ChangeDF


# In[154]:


ChangeDF.iloc[0,0:10]


# ## 8. Fazit

# Währungen untereinander in der Regel stark abhängig.  
# Auch innerhalb einer Währung starke Korrelation der verschiedenen ValueCategory (Preis, Marktkapitalisierung, 24 Volume, 24h Wachstum)
# Korrelation zu % pos oder neg Tweets meistens nicht wirklich erkennbar, gibt einige Fälle, in denen Abhängigkeit hineininterpretiert werden kann, aber könnte sich auch nur um Zufall handeln
# Mögliche Gründe: es gibt wirklich keine Abhängigkeit **oder** verwendete Bibliothek TextBlob zur Sentiment Analysis performt nicht gut. Gesamte Analysen bzgl. positiver und negativer Tweets basieren auf Klassifizierung der Bibliothek. In einem Anwendungsfall mit echtem dahinterstehenden Bedarf hätte noch mehr Zeit in Verifizierung und Auswahl der Sentiment Analysis Bibliothek gesteckt werden müssen, hätte jedoch Rahmen des Projekts gesprengt

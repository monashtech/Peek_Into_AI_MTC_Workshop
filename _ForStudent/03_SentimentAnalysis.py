# nltk library
import nltk
# do this once onnly do download the lexicon
# note: vader is tuned for social media text (ps: can handle emoji as well!)
# VADER = Valence Aware Dictionary and sEntiment Reasoner
nltk.download('vader_lexicon')

def obtain_sentiment(current_analyzer, current_comment):    
    current_sentiment = current_analyzer.polarity_scores(current_comment)
    print(current_comment)
    print(current_sentiment)

if __name__ == "__main__":
    # set to current working directory
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)  
    
    # load the analyzer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    current_analyzer = SentimentIntensityAnalyzer()
    
    obtain_sentiment(current_analyzer, "This movie is good.")
    obtain_sentiment(current_analyzer, "This movie is good!")
    obtain_sentiment(current_analyzer, "This movie is marginally good.")
    obtain_sentiment(current_analyzer, "This movie is extremely good.")

    current_comment = ""
    while current_comment != "quit":
        print("--------------------------------")
        current_comment = input("Type in your sentence (enter quit to quit): ")
        obtain_sentiment(current_analyzer, current_comment)

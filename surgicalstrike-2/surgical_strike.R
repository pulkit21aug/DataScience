install.packages("tm", dependencies=TRUE)
install.packages("twitteR" ,dependencies =  TRUE)
install.packages("wordcloud")
install.packages("ROAuth" , dependencies = TRUE)

install.packages("syuzhet")

library(twitteR)
library("tm")
library("wordcloud")
library(syuzhet)
library(ggplot2)
library(dplyr)


consumer_key <- 'RAMAho3sMbXZCSqoDPIF89Yym'
consumer_secret <- 'X0iBRYryXQuLfDMHB57cQPMi6zWRtvpsEWmybbyKHAYy2DrTM3'
access_token <- '1091957292130459648-RipEBIgDJmr4tmdbz1b7fRVi5MzPf8'
access_secret <- 'vWftc7oJHtge229RkwZMJHAJusV3rtlSMIhuWYn8vH0DO'
setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)


#pulwama attack 
balakot <- searchTwitter("#balakot", n=5000, lang="en") 
IndiaStrikesBack <- searchTwitter("#IndiaStrikesBack", n=5000, lang="en")
surgicalstrike2 <- searchTwitter("#surgicalstrike2", n=5000,lang="en") 

#merge twiiter data
final_data <- append(balakot,pulwama_2)
final_data <- append(final_data,IndiaStrikesBack)
final_data <- append(final_data,surgicalstrike2)

testdata <- twListToDF(final_data)

write.csv(testdata,file="tweets.csv")

#convert all text to lower case
testdata_text <- tolower(testdata$text)

View(testdata_text)


# Replace blank space (“rt”)
testdata_text <- gsub("rt", "", testdata_text)
# Replace @UserName
testdata_text <- gsub("@\\w+", "", testdata_text)

# Remove punctuation
testdata_text <- gsub("[[:punct:]]", "", testdata_text)

# Remove links
testdata_text <- gsub("http\\w+", "", testdata_text)

# Remove tabs
testdata_text <- gsub("[ |\t]{2,}", "", testdata_text)

# Remove blank spaces at the beginning
testdata_text <- gsub("^ ", "", testdata_text)

# Remove blank spaces at the end
testdata_text <- gsub(" $", "", testdata_text)


#corpus build - remove stop words
testdata_text_corpus <- Corpus(VectorSource(testdata_text))
testdata_text_corpus <- tm_map(testdata_text_corpus, function(x)removeWords(x,stopwords()))

?wordcloud

wordcloud(testdata_text_corpus,min.freq = 500,colors=brewer.pal(8, "Dark2"),random.color = TRUE,max.words = 15000)

#what people are saying about modi
testdata_text_tdm <- TermDocumentMatrix(testdata_text_corpus)
findAssocs(testdata_text_tdm, 'modi',.50)
##paies      board immaterial confidence      forge     natio…       view 
##0.60       0.57       0.57       0.56       0.56       0.56       0.56 

install.packages("qdap")
library(qdap)

#word_associate

word_associate(
  testdata_text,
  match.string = c("modi"),
  network.plot = T,
  wordcloud = TRUE,
  cloud.colors = c("gray85", "darkred")
)


#sentiment analysis
testdata_text_sent<-get_nrc_sentiment((testdata_text))

#calculationg total score for each sentiment
testdata_text_sent_score<-data.frame(colSums(testdata_text_sent[,]))

names(testdata_text_sent_score)<-"Score"
testdata_text_sent_score<-cbind("sentiment"=rownames(testdata_text_sent_score),testdata_text_sent_score)
rownames(testdata_text_sent_score)<-NULL


#plotting the sentiments with scores
ggplot(data=testdata_text_sent_score,aes(x=sentiment,y=Score))+geom_bar(aes(fill=sentiment),stat = "identity")+
  theme(legend.position="none")+
  xlab("Sentiments")+ylab("scores")+ggtitle("Sentiments of people ")


#remove positive , negative score 
testdata_text_sent<-get_nrc_sentiment((testdata_text))

testdata_text_sent_no_pos_neg<-select(testdata_text_sent,anger,anticipation,disgust,joy,sadness,surprise,trust)

#calculationg total score for each sentiment
testdata_text_sent_no_pos_neg<-data.frame(colSums(testdata_text_sent_no_pos_neg[,]))

names(testdata_text_sent_no_pos_neg)<-"Score"
testdata_text_sent_no_pos_neg<-cbind("sentiment"=rownames(testdata_text_sent_no_pos_neg),testdata_text_sent_no_pos_neg)
rownames(testdata_text_sent_no_pos_neg)<-NULL


#plotting the sentiments with scores
ggplot(data=testdata_text_sent_no_pos_neg,aes(x=sentiment,y=Score))+geom_bar(aes(fill=sentiment),stat = "identity")+
  theme(legend.position="none")+
  xlab("Sentiments")+ylab("scores")+ggtitle("Sentiments of people ")
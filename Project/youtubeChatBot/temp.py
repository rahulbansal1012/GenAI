from youtube_transcript_api import YouTubeTranscriptApi ,TranscriptsDisabled , FetchedTranscriptSnippet

# video_id = 'J5_-l7WIO_w'
ytt_api = YouTubeTranscriptApi()
fetched = ytt_api.fetch(
    video_id='J5_-l7WIO_w' , 
       languages=("hi", "en")                 )
print("TYPE:: " , type(fetched))
fetched2 =  YouTubeTranscriptApi.get_transcript("J5_-l7WIO_w")
print("Content Return:" , fetched2)



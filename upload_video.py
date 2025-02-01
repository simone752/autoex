import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Authenticate
def get_authenticated_service():
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        "client_secret.json", SCOPES)
    credentials = flow.run_console()
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)

# Upload a video
def upload_video(video_file, title):
    youtube = get_authenticated_service()
    request_body = {
        'snippet': {
            'title': title,
            'description': 'Generated mysterious video',
            'tags': ['creepy', 'mystery'],
            'categoryId': '24'
        },
        'status': {
            'privacyStatus': 'public'
        }
    }
    
    media_file = googleapiclient.http.MediaFileUpload(video_file)
    
    request = youtube.videos().insert(
        part="snippet,status",
        body=request_body,
        media_body=media_file
    )
    response = request.execute()
    print(f"Video uploaded: {response['id']}")

if __name__ == "__main__":
    for filename in os.listdir("generated_videos"):
        if filename.endswith(".mp4"):
            upload_video(os.path.join("generated_videos", filename), "Mysterious Video Title")
            os.remove(os.path.join("generated_videos", filename))

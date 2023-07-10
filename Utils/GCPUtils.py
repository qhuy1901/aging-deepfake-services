import uuid
from google.cloud import storage
import datetime

class GCPUtils:
    def save_image_to_gif_file_to_gcloud(source_file_name):
        client = storage.Client.from_service_account_json('credentials.json')

        # Set the name of the bucket and the path to the file to upload
        bucket_name = 'aging-image-video-storage'

        current_time = datetime.now()
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        destination_blob_name = 'output/' + time_string + str(uuid.uuid4()) + ".gif"

        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        # Print the URL of the uploaded file
        print('File uploaded to: {}'.format(blob.public_url))
        return blob.public_url
    

    def save_portrait_to_gcs(self, portrait):
        # Create a client instance for GCS
        client = storage.Client.from_service_account_json('credentials.json')

        # Get the bucket object
        bucket = client.bucket('aging-output')

        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        destination_blob_name = 'output/portrait/' + time_string + str(uuid.uuid4()) + ".jpg"

        # Create a blob object with the desired filename
        blob = bucket.blob(destination_blob_name)

        # Set the content type of the blob
        blob.content_type = 'image/jpg'

        # Upload the portrait image to GCS
        blob.upload_from_file(portrait)

        # Generate a signed URL for the uploaded image
        signed_url = blob.generate_signed_url(
            version='v4',
            expiration=datetime.timedelta(minutes=15),
            method='GET'
        )

        return signed_url

    def upload_video(self, video):
        # Create a client instance for GCS
        client = storage.Client.from_service_account_json('credentials.json')

        # Get the bucket object
        bucket = client.bucket('aging-image-video-storage')

        current_time = datetime.datetime.now()
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        destination_blob_name = 'output/video/' + time_string + str(uuid.uuid4()) + ".mp4"

         # Create a blob object with the desired filename
        blob = bucket.blob(destination_blob_name)

        # Set the content type of the blob
        blob.content_type = 'video/mp4'

        # Open the video file in read mode
        with open(video, 'rb') as video_file:
            # Upload the video file to GCS
            blob.upload_from_file(video_file)

        # Generate a signed URL for the uploaded image
        signed_url = blob.generate_signed_url(
            version='v4',
            expiration=datetime.timedelta(minutes=15),
            method='GET'
        )

        return signed_url
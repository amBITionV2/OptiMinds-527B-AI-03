import firebase_admin
from firebase_admin import credentials, firestore
import base64
import numpy as np
import cv2

# Initialize Firebase Admin SDK (Make sure serviceAccountKey.json exists)
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Firestore Client
db = firestore.client()

# Function to Retrieve and Process Images


def fetch_initial_images(unique_id):
    """Fetch the first 5 images for a specific user from Firestore and store their embeddings."""
    initial_face_embeddings = []  # Store face embeddings (if needed)

    doc_ref = db.collection("images").document(
        str(unique_id))  # Ensure ID is a string

    try:
        # Force Firestore to fetch fresh data (ignore cache)
        doc = doc_ref.get({"source": "server"})

        if not doc.exists:
            print(f"❌ No images found for user: {unique_id}")
            return []

        data = doc.to_dict()
        # Debugging: Print full document
        print(f"✅ Firestore Data for {unique_id}: {data}")

        # Determine where the images are stored in Firestore
        if "images" in data:
            image_list = data["images"]
        elif "default" in data and "images" in data["default"]:
            image_list = data["default"]["images"]
        else:
            print("❌ Could not find images in document.")
            return []

        # Debugging: Check image count
        print(f"🔍 Found {len(image_list)} images for user: {unique_id}")

        # Process only the first 5 images
        image_list = image_list[:5]
        processed_images = []

        for index, image_data in enumerate(image_list):
            try:
                # Decode Base64 image
                image_bytes = base64.b64decode(image_data)
                image_array = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f"❌ Image {index+1}: Decoding failed!")
                    continue

                processed_images.append(frame)  # Store valid images
                print(f"✅ Image {index+1}: Successfully processed!")

            except Exception as e:
                print(f"❌ Image {index+1}: Error - {e}")

        print(f"✅ Successfully retrieved {len(processed_images)} images.")

        return processed_images  # Return list of processed images

    except Exception as e:
        print(f"❌ Error fetching images: {e}")
        return []


# Example Usage
if __name__ == "__main__":
    user_id = input("Enter User ID: ")
    images = fetch_initial_images(user_id)
    print(f"Total Images Retrieved: {len(images)}")

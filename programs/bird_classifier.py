import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model_path = "Programs/bird_classifier_model_2.h5"  # Replace with the path to your trained model
model = load_model(model_path)

# Define the class labels
class_labels = ["Asian Green Bee-Eater", "Common Kingfisher", "Coppersmith Barbet", "Gray Wagtail",
                "Indian Pitta", "Indian Roller", "Ruddy Shelduck"]

# Define the video file path  Replace with the path to your input video
# video_path = "Test_Video\Asian_green _bee-eater_4.mp4"
# video_path = "Test_Video\Common-Kingfisher.mp4"
# video_path = "Test_Video\Coppersmith-barbet_2.mp4"
# video_path = "Test_Video\Grey-wagtail_3.mp4"
video_path = "Test_Video\Indian-pitta_3.mp4"
# video_path = "Test_Video\Indian-Roller_3.mp4"
# video_path = "Test_Video\Ruddy-shelduck_2.mp4"

# Define the file paths for bird information
bird_info_files = {
    "Asian Green Bee-Eater": "bird_info\The Asian green bee-eater.txt",
    "Common Kingfisher": "bird_info\The common kingfisher.txt",
    "Coppersmith Barbet": "bird_info\The coppersmith barbet.txt",
    "Gray Wagtail": "bird_info\grey wagtail.txt",
    "Indian Pitta": "bird_info\The Indian pitta.txt",
    "Indian Roller": "bird_info\indian roller.txt",
    "Ruddy Shelduck": "bird_info\The ruddy shelduck.txt"
}

# Load bird information from text files
bird_info = {}
for label, file_path in bird_info_files.items():
    with open(file_path, "r") as info_file:
        bird_info[label] = info_file.read()

# Open the video file
video = cv2.VideoCapture(video_path)

# Get the video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
output_path = "output_video_7.mp4"  # Replace with the desired output video path
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    # Read the frame
    ret, frame = video.read()

    if not ret:
        break

    # Preprocess the frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Perform the classification
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]
    confidence = np.max(predictions[0]) * 100

    # Get the coordinates for the bounding box
    height, width, _ = frame.shape
    start_x = int(width * 0.2)
    start_y = int(height * 0.2)
    end_x = int(width * 0.8)
    end_y = int(height * 0.8)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    # Draw the predicted label and confidence on the frame
    label_text = f"{predicted_label}: {confidence:.2f}%"
    cv2.putText(frame, label_text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame to the output video
    output_video.write(frame)

    # Display the frame with the bounding box and predicted label
    cv2.imshow("Bird Classification", frame)

    

    # Wait for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Predicted Bird:", predicted_label)

# Print bird information
if predicted_label in bird_info:
    print("Bird Information:")
    print(bird_info[predicted_label])

# Release the video capture and writer
video.release()
output_video.release()

# Close all windows
cv2.destroyAllWindows()

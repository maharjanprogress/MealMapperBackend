from flask import Flask, jsonify, request
from sqlalchemy.exc import IntegrityError
from config import Config
from models import db, User,Challenge,AcceptedChallenge,LeaderBoard
from datetime import date,datetime
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)



def create_response(status, code, message, details=None):
    """
    Helper function to create a consistent JSON response format.
    """
    return jsonify({
        "status": status,
        "code": code,
        "message": message,
        "details": details
    }), code


@app.route('/login', methods=['POST'])
def login():
    # Check if the request is JSON
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    # Parse the JSON request body
    data = request.get_json()

    # Extract username and password from the request
    username = data.get('username')
    password = data.get('password')

    # Validate input
    if not username or not password:
        return create_response("error", 400, "Username and password are required")

    try:
        # Query the database for the user
        user = User.query.filter_by(username=username).first()

        # Check if the user exists and the password matches
        if user and user.password == password:
            return create_response("success", 200, "Login successful", {"username": user.username, "userId": user.userid, "userEmail": user.email})
        else:
            return create_response("error", 401, "Credentials don't match")

    except Exception as e:
        return create_response("error", 500, "An error occurred during login", str(e))

@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return create_response("success", 200, "Users retrieved successfully", [user.to_dict() for user in users])

@app.route('/users', methods=['POST'])
def add_user():
    # Check if the request is JSON
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    # Parse the JSON request body
    data = request.get_json()

    # Extract individual parameters from the JSON body
    password = data.get('password')
    email = data.get('email')
    username = data.get('username')
    joinDate = date.today()
    age = data.get('age')
    gender = data.get('gender')
    height_cm = data.get('height_cm')
    weight_kg = data.get('weight_kg')
    activity_level = data.get('activity_level')
    goal = data.get('goal')
    dietary_pref = data.get('dietary_pref')
    allergies = data.get('allergies')  # List of strings
    medical_conditions = data.get('medical_conditions')  # List of strings
    meal_times = data.get('meal_times')  # JSON object
    address = data.get('address')

    # Validate required fields
    if not username or not email or not password:
        return create_response("error", 400, "Username, email, and password are required")

    # Map gender, activity_level, and goal to numeric values
    gender_map = {'Male': 1, 'Female': -1, 'Other': 0}
    activity_level_map = {'Not Active': -1, 'Moderate': 0, 'Very Active': 1}
    goal_map = {'Go Slim': -1, 'Maintain': 0, 'Gain Weight': 1}

    gender = gender_map.get(gender)
    activity_level = activity_level_map.get(activity_level)
    goal = goal_map.get(goal)

    # Validate mapped values
    if gender is None or activity_level is None or goal is None:
        return create_response("error", 400, "Invalid value for gender, activity_level, or goal")

    try:
        # Create a new User object
        new_user = User(
            password=password,
            email=email,
            username=username,
            age=age,
            gender=gender,
            height_cm=height_cm,
            weight_kg=weight_kg,
            activity_level=activity_level,
            goal=goal,
            dietary_pref=dietary_pref,
            allergies=allergies,
            medical_conditions=medical_conditions,
            meal_times=meal_times,
            joindate=joinDate,
            address=address
        )
        db.session.add(new_user)  # Add the new user to the session
        db.session.commit()  # Commit the transaction to save the user
        return create_response("success", 201, "User added successfully", new_user.to_dict())

    except IntegrityError:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 400, "Username or email already exists")

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))

@app.route('/challenges', methods=['GET'])
def get_challenge():
    challenges = Challenge.query.all()
    return create_response("success", 200, "Challenges retrieved successfully", [challenge.to_dict() for challenge in challenges])


@app.route('/challenges', methods=['POST'])
def add_challenge():
    # Extract parameters from the request
    title = request.form.get('title')
    description = request.form.get('description')
    reward_points = request.form.get('reward_points', type=int)
    deadline = request.form.get('deadline')  # Expected in ISO format (e.g., "2025-04-20T23:59:59")

    # Validate required fields
    if not title or not reward_points or not deadline:
        return create_response("error", 400, "Title, reward points, and deadline are required")

    try:
        # Convert deadline to a datetime object
        deadline = datetime.fromisoformat(deadline)

        # Create a new Challenge object
        new_challenge = Challenge(
            title=title,
            description=description,
            reward_points=reward_points,
            deadline=deadline
        )
        db.session.add(new_challenge)  # Add the new challenge to the session
        db.session.commit()  # Commit the transaction to save the challenge
        return create_response("success", 201, "Challenge added successfully", new_challenge.to_dict())

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))


@app.route('/accepted_challenges', methods=['GET'])
def get_accepted_challenges():
    accepted_challenges = AcceptedChallenge.query.all()
    return create_response("success", 200, "Accepted challenges retrieved successfully", [challenge.to_dict() for challenge in accepted_challenges])


@app.route('/accepted_challenges', methods=['POST'])
def add_accepted_challenge():
    # Extract parameters from the request
    challengeId = request.form.get('challengeId', type=int)
    userId = request.form.get('userId', type=int)
    progress = request.form.get('progress', type=int, default=0)
    deadline_reached = request.form.get('deadline_reached', type=bool, default=False)
    accepted_date = request.form.get('accepted_date')  # Optional, in ISO format

    # Validate required fields
    if not challengeId or not userId:
        return create_response("error", 400, "Challenge ID and User ID are required")

    try:
        # Convert accepted_date to a datetime object if provided
        accepted_date = datetime.fromisoformat(accepted_date) if accepted_date else datetime.utcnow()

        # Create a new AcceptedChallenge object
        new_accepted_challenge = AcceptedChallenge(
            challengeId=challengeId,
            userId=userId,
            progress=progress,
            deadline_reached=deadline_reached,
            accepted_date=accepted_date
        )
        db.session.add(new_accepted_challenge)  # Add the new accepted challenge to the session
        db.session.commit()  # Commit the transaction to save the accepted challenge
        return create_response("success", 201, "Accepted challenge added successfully", new_accepted_challenge.to_dict())

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))


@app.route('/leader_board', methods=['GET'])
def get_leader_board():
    leader_board_entries = LeaderBoard.query.all()
    return create_response("success", 200, "Leaderboard entries retrieved successfully", [entry.to_dict() for entry in leader_board_entries])


@app.route('/leader_board', methods=['POST'])
def add_leader_board_entry():
    # Extract parameters from the request
    userId = request.form.get('userId', type=int)
    season = request.form.get('season')
    points = request.form.get('points', type=int, default=0)
    last_updated_date = request.form.get('last_updated_date')  # Optional, in ISO format

    # Validate required fields
    if not userId or not season:
        return create_response("error", 400, "User ID and season are required")

    try:
        # Convert last_updated_date to a datetime object if provided
        last_updated_date = datetime.fromisoformat(last_updated_date) if last_updated_date else datetime.utcnow()

        # Create a new LeaderBoard entry
        new_leader_board_entry = LeaderBoard(
            userId=userId,
            season=season,
            points=points,
            last_updated_date=last_updated_date
        )
        db.session.add(new_leader_board_entry)  # Add the new leaderboard entry to the session
        db.session.commit()  # Commit the transaction to save the leaderboard entry
        return create_response("success", 201, "Leaderboard entry added successfully", new_leader_board_entry.to_dict())

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))

@app.route('/get_image', methods=['POST'])
def handle_image():
    import traceback
    from PIL import Image
    # Define the directory where the image will be temporarily saved
    save_directory = r"C:\Users\progr\OneDrive\Desktop\MealMapperBackend"

    # Check if the request contains a file
    if 'image' not in request.files:
        return create_response("error", 400, "No image file provided")

    image = request.files['image']

    # Check if the file has a valid name
    if image.filename == '':
        return create_response("error", 400, "No selected file")

    try:
        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(image.filename)

        # Save the image temporarily
        file_path = os.path.join(save_directory, filename)
        
        # Resize the image to 64x64 before saving
        img = Image.open(image)
        img.save(file_path)

        # Limit TensorFlow GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # Load the model and make predictions
        cnn = tf.keras.models.load_model(r"C:\Users\progr\OneDrive\Desktop\MLdataset\trained_model1.h5")
        imageFinal = tf.keras.preprocessing.image.load_img(file_path, target_size=(64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(imageFinal)
        input_arr = np.array([input_arr])  # Convert single image into batch (2D Array)
        predictions = cnn.predict(input_arr)
        result_index = np.argmax(predictions)

        # Reading Labels
        with open(r"C:\Users\progr\OneDrive\Desktop\MLdataset\labels.txt") as f:
            content = f.readlines()
        label = [i.strip() for i in content]
        result = label[result_index]

        # Delete the image after saving
        os.remove(file_path)

        return create_response("success", 200, "Image processed and deleted successfully", result)

    except Exception as e:
        print(traceback.format_exc())  # Log the full traceback
        return create_response("error", 500, "An error occurred while processing the image", str(e))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # only creates tables if they don’t exist
    app.run(debug=True,port=8000)

from flask import Flask, jsonify, request
from flask import Flask, send_from_directory
from sqlalchemy.exc import IntegrityError
from config import Config
from models import db, User,Challenge,AcceptedChallenge,LeaderBoard,Food,ScannedHistory
from datetime import date,datetime,timedelta
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

@app.route('/challenges/<int:userId>', methods=['GET'])
def get_challenges_for_user(userId):
    today = datetime.today()

    # Get all challenge IDs that the user has already accepted
    accepted_challenge_ids = db.session.query(AcceptedChallenge.challengeId).filter_by(userId=userId).all()
    accepted_challenge_ids = [row[0] for row in accepted_challenge_ids]  # Extract IDs from query result
    # Query challenges where the deadline is in the future and not already accepted by the user
    challenges = Challenge.query.filter(
        Challenge.deadline > today,
        ~Challenge.challengeId.in_(accepted_challenge_ids)  # Exclude accepted challenges
    ).all()

    # Format the response
    formatted_challenges = []
    for challenge in challenges:
        duration = (challenge.deadline - today).days  # Calculate duration in days
        formatted_challenges.append({
            'id': challenge.challengeId,
            'title': challenge.title,
            'description': challenge.description,
            'difficulty': challenge.difficulty,
            'points': challenge.reward_points,
            'duration': f"{duration} days" if duration > 0 else "Expired"
        })

    return create_response("success", 200, "Challenges retrieved successfully", formatted_challenges)



@app.route('/challenges', methods=['POST'])
def add_challenge():
    # Check if the request is JSON
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    # Parse the JSON request body
    data = request.get_json()

    # Extract parameters from the JSON body
    title = data.get('title')
    description = data.get('description')
    reward_points = data.get('reward_points')
    deadline = data.get('deadline')  # Expected in ISO format (e.g., "2025-04-29T23:59:59")
    difficulty = data.get('difficulty')  # New field
    requirements = data.get('requirements')  # New field


    # Validate required fields
    if not title or not reward_points or not deadline or not requirements:
        return create_response("error", 400, "Title, reward points, and deadline are required")

    try:
        # Convert deadline to a datetime object
        deadline = datetime.fromisoformat(deadline)

        # Create a new Challenge object
        new_challenge = Challenge(
            title=title,
            description=description,
            reward_points=reward_points,
            deadline=deadline,
            difficulty=difficulty,
            requirements=requirements

        )
        db.session.add(new_challenge)  # Add the new challenge to the session
        db.session.commit()  # Commit the transaction to save the challenge
        return create_response("success", 201, "Challenge added successfully", new_challenge.to_dict())

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))


@app.route('/accepted_challenges/<int:userId>', methods=['GET'])
def get_accepted_challenges_for_user(userId):
    today = datetime.today()
    print(today)
    # Perform an inner join between AcceptedChallenge and Challenge
    results = db.session.query(
        AcceptedChallenge.progress,
        AcceptedChallenge.completed,
        Challenge.title,
        Challenge.description,
        Challenge.deadline,
        Challenge.reward_points
    ).join(Challenge, AcceptedChallenge.challengeId == Challenge.challengeId) \
     .filter(
         AcceptedChallenge.userId == userId,  # Filter by userId
         AcceptedChallenge.completed == False,  # Only show incomplete challenges
         Challenge.deadline > today  # Only show challenges whose deadline has not passed
     ).all()

    # Format the response
    formatted_challenges = []
    for result in results:
        progress, completed, title, description, deadline, reward_points = result
        days_left = (deadline - today).days  # Calculate days left until the deadline
        formatted_challenges.append({
            'title': title,
            'description': description,
            'progress': progress,
            'deadline': deadline.strftime('%Y-%m-%d %H:%M'),  # Format deadline
            'points': reward_points,
            'daysLeft': days_left
        })

    return create_response("success", 200, "Accepted challenges retrieved successfully", formatted_challenges)


@app.route('/completed_challenges/<int:userId>', methods=['GET'])
def get_completed_challenges_for_user(userId):
    # Perform an inner join between AcceptedChallenge and Challenge
    results = db.session.query(
        Challenge.title,
        Challenge.description,
        Challenge.deadline,
        Challenge.reward_points
    ).select_from(AcceptedChallenge).join(Challenge, AcceptedChallenge.challengeId == Challenge.challengeId) \
     .filter(
         AcceptedChallenge.userId == userId,  # Filter by userId
         AcceptedChallenge.completed == True  # Only show completed challenges
     ).all()

    # Format the response
    formatted_challenges = []
    for result in results:
        title, description, deadline, reward_points = result
        formatted_challenges.append({
            'title': title,
            'description': description,
            'completedOn': deadline.strftime('%d %b %Y'),  # Format deadline as '20 Apr 2025'
            'points': reward_points
        })

    return create_response("success", 200, "Completed challenges retrieved successfully", formatted_challenges)



@app.route('/incompleted_challenges/<int:userId>', methods=['GET'])
def get_incompleted_challenges_for_user(userId):
    today = datetime.today()
    # Perform an inner join between AcceptedChallenge and Challenge
    results = db.session.query(
        Challenge.title,
        Challenge.description,
        Challenge.deadline,
        Challenge.reward_points
    ).join(Challenge, AcceptedChallenge.challengeId == Challenge.challengeId) \
     .filter(
         AcceptedChallenge.userId == userId,  # Filter by userId
         Challenge.deadline < today,
         AcceptedChallenge.completed == False  # Only show completed challenges
     ).all()

    # Format the response
    formatted_challenges = []
    for result in results:
        title, description, deadline, reward_points = result
        formatted_challenges.append({
            'title': title,
            'description': description,
            'completedOn': deadline.strftime('%d %b %Y'),  # Format deadline as '20 Apr 2025'
            'points': reward_points
        })

    return create_response("success", 200, "Completed challenges retrieved successfully", formatted_challenges)


@app.route('/accepted_challenges', methods=['POST'])
def add_accepted_challenge():
    # Check if the request is JSON
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    # Parse the JSON request body
    data = request.get_json()

    # Extract parameters from the JSON body
    challengeId = data.get('challengeId')
    userId = data.get('userId')

    # Validate required fields
    if not challengeId or not userId:
        return create_response("error", 400, "Challenge ID and User ID are required")

    try:
        # Set default values
        progress = 0
        completed = False
        accepted_date = datetime.today()  # Use today's date in UTC

        # Create a new AcceptedChallenge object
        new_accepted_challenge = AcceptedChallenge(
            challengeId=challengeId,
            userId=userId,
            progress=progress,
            completed=completed,
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
        last_updated_date = datetime.fromisoformat(last_updated_date) if last_updated_date else datetime.today()

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
    save_directory = r""

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
        cnn = tf.keras.models.load_model(r"trained_model1.h5")
        imageFinal = tf.keras.preprocessing.image.load_img(file_path, target_size=(64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(imageFinal)
        input_arr = np.array([input_arr])  # Convert single image into batch (2D Array)
        predictions = cnn.predict(input_arr)
        result_index = np.argmax(predictions)

        # Reading Labels
        with open(r"labels.txt") as f:
            content = f.readlines()
        label = [i.strip() for i in content]
        result = label[result_index]

        # Delete the image after saving
        os.remove(file_path)

        return create_response("success", 200, "Image processed and deleted successfully", result)

    except Exception as e:
        print(traceback.format_exc())  # Log the full traceback
        return create_response("error", 500, "An error occurred while processing the image", str(e))


@app.route('/foods', methods=['POST'])
def add_food():
    # Check if the request is JSON
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    # Parse the JSON request body
    data = request.get_json()

    # Extract parameters from the JSON body
    name = data.get('name')
    description = data.get('description')
    calories_per_serving = data.get('calories_per_serving')
    protein_per_serving = data.get('protein_per_serving', 0)
    carbs_per_serving = data.get('carbs_per_serving', 0)
    fat_per_serving = data.get('fat_per_serving', 0)
    sugar_per_serving = data.get('sugar_per_serving', 0)
    serving_size = data.get('serving_size', 100)
    category = data.get('category')
    meal_type = data.get('meal_type')  # Required field
    tags = data.get('tags')  # JSON object
    contents = data.get('contents')  # JSON object
    recipe = data.get('recipe')  # JSON object
    image_url = data.get('image_url')
    popularity_score = data.get('popularity_score', 0)

    # Validate required fields
    if not name or not calories_per_serving or not meal_type:
        return create_response("error", 400, "Name, calories per serving, and meal type are required")

    # Validate meal_type
    valid_meal_types = ['breakfast', 'lunch', 'dinner']
    if meal_type not in valid_meal_types:
        return create_response("error", 400, f"Invalid meal type. Must be one of {valid_meal_types}")

    try:
        # Create a new Food object
        new_food = Food(
            name=name,
            description=description,
            calories_per_serving=calories_per_serving,
            protein_per_serving=protein_per_serving,
            carbs_per_serving=carbs_per_serving,
            fat_per_serving=fat_per_serving,
            sugar_per_serving=sugar_per_serving,
            serving_size=serving_size,
            category=category,
            meal_type=meal_type,
            tags=tags,
            contents=contents,
            recipe=recipe,
            image_url=image_url,
            popularity_score=popularity_score
        )
        db.session.add(new_food)  # Add the new food to the session
        db.session.commit()  # Commit the transaction to save the food
        return create_response("success", 201, "Food added successfully", new_food.to_dict())

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))

@app.route('/foods/', defaults={'foodname': None})
@app.route('/foods/<path:foodname>', methods=['GET'])
def get_food_by_name(foodname):
    try:

        if not foodname:
            foods = Food.query.all()
            formatted_foods = [food.to_dictFind() for food in foods]
            return create_response(
                "success", 
                200, 
                f"Retrieved all {len(foods)} foods",
                formatted_foods
            )
        
        # Convert foodname to lowercase for case-insensitive search
        search_term = foodname.lower()
        
        # Query foods where name contains the search term
        foods = Food.query.filter(Food.name.ilike(f'%{search_term}%')).all()
        
        if not foods:
            return create_response("error", 404, f"No foods found matching '{foodname}'")
        
        # Format the response
        formatted_foods = [food.to_dictFind() for food in foods]
        
        return create_response(
            "success", 
            200, 
            f"Found {len(foods)} food(s) matching '{foodname}'",
            formatted_foods
        )

    except Exception as e:
        return create_response("error", 500, "An error occurred while retrieving food information", str(e))


@app.route('/scanned_food/<path:userid>', methods=['POST'])
def add_scanned_food(userid):
    try:
        if not request.is_json:
            return create_response("error", 415, "Content-Type must be application/json")

        data = request.get_json()
        food_name = data.get('name')
        serving_size = data.get('serving_size', 100.0)
        meal_type = data.get('meal_type', '').lower()

        if not food_name:
            return create_response("error", 400, "Food name is required")

        # If meal_type is empty, determine it based on current time
        if not meal_type:
            current_hour = datetime.now().hour
            if 5 <= current_hour < 11:
                meal_type = 'breakfast'
            elif 11 <= current_hour < 16:
                meal_type = 'lunch'
            else:
                meal_type = 'dinner'
        else:
            # Validate meal_type if provided
            valid_meal_types = ['breakfast', 'lunch', 'dinner']
            if meal_type not in valid_meal_types:
                return create_response("error", 400, f"Invalid meal type. Must be one of {valid_meal_types}")

        # Find the food by name
        food = Food.query.filter(Food.name.ilike(food_name.lower())).first()
        if not food:
            return create_response("error", 404, f"Food '{food_name}' not found in database")
        
        # Check for duplicate entries within last 5 minutes
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        recent_scan = ScannedHistory.query.filter(
            ScannedHistory.userId == userid,
            ScannedHistory.foodId == food.foodId,
            ScannedHistory.scanned_at >= five_minutes_ago
        ).first()

        if recent_scan:
            return create_response(
                "error", 
                409, 
                "Duplicate scan detected. Please wait 5 minutes before scanning the same food again.",
                None
            )
        
        # Create new scanned history entry
        new_scan = ScannedHistory(
            userId=userid,
            foodId=food.foodId,
            meal_time=meal_type,
            servings=serving_size,  # Default to 1 serving
            scanned_at=datetime.today()
        )

        # Add and commit to database
        db.session.add(new_scan)
        db.session.commit()

        return create_response(
            "success", 
            201, 
            "Food scan recorded successfully",
            new_scan.to_dict()
        )

    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while recording food scan", str(e))


@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory('FoodImage', filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # only creates tables if they don’t exist
    app.run(debug=True,port=8000)

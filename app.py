from flask import Flask, jsonify, request
from flask import Flask, send_from_directory
from sqlalchemy.exc import IntegrityError
from config import Config
from models import db, User,Challenge,AcceptedChallenge,LeaderBoard,Food,ScannedHistory, Ingredient, Allergen, MedicalConditionDietaryGuideline, ingredient_allergen_association, FoodItemIngredient, UserFoodInteraction
from sqlalchemy import func
from datetime import date,datetime,timedelta
from dateutil.relativedelta import relativedelta
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import json

app = Flask(__name__)
app.config.from_object(Config)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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

def normalize_string_list(input_list):
    """
    Normalizes a list of strings:
    - Converts items to lowercase and strips whitespace.
    - Removes "none" (case-insensitive) and empty strings.
    - Returns an empty list if input is None, not a list, 
      or results in an empty list after processing.
    """
    if not isinstance(input_list, list):
        return []  # Return empty list if input is not a list or None

    processed_list = []
    for item in input_list:
        if isinstance(item, str):
            cleaned_item = item.lower().strip()
            if cleaned_item and cleaned_item != 'none':  # Add if not empty and not "none"
                processed_list.append(cleaned_item)
        # else:
            # Optionally, log or handle non-string items if they are unexpected
            # print(f"Warning: Non-string item '{item}' found in list, skipping.")
    return processed_list

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
    dietary_pref_raw = data.get('dietary_pref') # Defaulting later
    allergies_raw = data.get('allergies')  # List of strings
    medical_conditions_raw = data.get('medical_conditions')  # List of strings
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

    # Normalize and set defaults
    allergies = normalize_string_list(allergies_raw)
    medical_conditions = normalize_string_list(medical_conditions_raw)
    dietary_pref = (dietary_pref_raw.lower().strip() if isinstance(dietary_pref_raw, str) and dietary_pref_raw.strip() else 'any')
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
        return create_response("error", 206, "Username or email already exists")

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))

@app.route('/allOngoingChallenge/', defaults={'challengeSearch': None}, methods=['GET'])
@app.route('/allOngoingChallenge/<path:challengeSearch>', methods=['GET'])
def get_all_challenges(challengeSearch):
    today = datetime.today()
    
    query = Challenge.query.filter(Challenge.deadline > today)

    if challengeSearch:
        search_term = f"%{challengeSearch.lower()}%"
        query = query.filter(
            db.or_(
                Challenge.title.ilike(search_term),
                Challenge.description.ilike(search_term)
            )
        )
    
    challenges = query.all()
    
    return create_response("success", 200, "Ongoing challenges retrieved successfully", [{'id': challenge.challengeId, 'title': challenge.title, 'description': challenge.description, 'difficulty': challenge.difficulty, 'reward_points': challenge.reward_points, 'deadline': challenge.deadline.isoformat()} for challenge in challenges])

@app.route('/challenge/<int:challengeId>', methods=['GET'])
def get_challenge_by_id(challengeId):
    challenge = Challenge.query.get(challengeId)
    if not challenge:
        return create_response("error", 404, "Challenge not found", None)
    return create_response("success", 200, "Challenge retrieved successfully", challenge.to_dict())

@app.route('/challenge/<int:challengeId>', methods=['PUT'])
def update_challenge(challengeId):
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    data = request.get_json()
    challenge = db.session.get(Challenge, challengeId)

    if not challenge:
        return create_response("error", 404, f"Challenge with ID {challengeId} not found")

    updates_made = False

    # Validate and update title
    if 'title' in data:
        new_title = data.get('title')
        if isinstance(new_title, str) and new_title.strip():
            challenge.title = new_title.strip()
            updates_made = True

    # Validate and update description
    if 'description' in data: # Description can be an empty string
        new_description = data.get('description')
        if isinstance(new_description, str):
            challenge.description = new_description
            updates_made = True

    # Validate and update reward_points
    if 'reward_points' in data:
        new_reward_points = data.get('reward_points')
        if isinstance(new_reward_points, int):
            challenge.reward_points = new_reward_points
            updates_made = True

    # Validate and update deadline
    if 'deadline' in data:
        new_deadline_str = data.get('deadline')
        try:
            challenge.deadline = datetime.fromisoformat(new_deadline_str)
            updates_made = True
        except (TypeError, ValueError):
            return create_response("error", 400, "Invalid deadline format. Expected ISO format (e.g., YYYY-MM-DDTHH:MM:SS)")

    # Validate and update difficulty
    if 'difficulty' in data: # Difficulty can be an empty string or None
        challenge.difficulty = data.get('difficulty')
        updates_made = True

    # Validate and update requirements (must be a dictionary/JSON object)
    if 'requirements' in data:
        new_requirements = data.get('requirements')
        if isinstance(new_requirements, dict):
            challenge.requirements = new_requirements
            updates_made = True

    if updates_made:
        try:
            db.session.commit()
            return create_response("success", 200, f"Challenge '{challenge.title}' updated successfully", challenge.to_dict())
        except Exception as e:
            db.session.rollback()
            return create_response("error", 500, "An error occurred while updating the challenge", str(e))
    else:
        return create_response("error", 400, "No valid fields provided for update or no changes made")

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
    ).select_from(AcceptedChallenge).join(Challenge, AcceptedChallenge.challengeId == Challenge.challengeId) \
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

@app.route('/foodsTrainedButNotInDB', methods=['GET'])
def get_foods_trained_but_not_in_db():
    foods = Food.query.all()
    food_names = [food.name for food in foods]
    # Read the labels from the file
    with open(r"labels.txt") as f:
        content = f.readlines()
    labels = [i.strip() for i in content]
    # Find foods that are in the labels but not in the database
    trained_but_not_in_db = [label for label in labels if label not in food_names]
    if not trained_but_not_in_db:
        return create_response("success", 200, "All trained foods are already in the database", [])
    
    if len(trained_but_not_in_db) == 1:
        return create_response("success", 200, "This Food is not in the database, Please add it", trained_but_not_in_db[0])
    return create_response("success", 200, "These "+str(len(trained_but_not_in_db))+" Foods are not in the database, Please add them", trained_but_not_in_db)

@app.route('/foods', methods=['POST'])
def add_food():
    # Expecting multipart/form-data
    if 'name' not in request.form: # Basic check for form data
        return create_response("error", 400, "Missing food name in form data.")

    data = request.form # Text data
    image_file = request.files.get('image') # Image file

    # Extract parameters from the JSON body
    name = data.get('name')
    name_stripped = name.strip().lower() if name else None
    description = data.get('description')
    calories_per_serving = data.get('calories_per_serving')
    protein_per_serving = data.get('protein_per_serving', 0)
    carbs_per_serving = data.get('carbs_per_serving', 0)
    fat_per_serving = data.get('fat_per_serving', 0)
    sugar_per_serving = data.get('sugar_per_serving', 0)
    sodium_per_serving = data.get('sodium_per_serving', 0)
    serving_size = data.get('serving_size', 100)
    category = data.get('category')
    meal_type = data.get('meal_type')  # Required field
    tags_str = data.get('tags')  # This is a JSON string like "[\"tag1\", \"tag2\"]"
    contents_str = data.get('contents')  # JSON string
    recipe_str = data.get('recipe')  # JSON string

    popularity_score = data.get('popularity_score', 0)

    # Validate required fields
    if not name_stripped or calories_per_serving is None or not meal_type or \
       sodium_per_serving is None or protein_per_serving is None or carbs_per_serving is None or \
       fat_per_serving is None or sugar_per_serving is None or serving_size is None:
        return create_response("error", 400, "Name, calories, meal type, serving size, and nutrient values (protein, carbs, fat, sugar) are required.")

    # Validate meal_type
    valid_meal_types = ['breakfast', 'lunch', 'dinner']
    if meal_type not in valid_meal_types:
        return create_response("error", 400, f"Invalid meal type. Must be one of {valid_meal_types}")

    # 1. Check if foodname already exists in the database (case-insensitive)
    existing_food = Food.query.filter(db.func.lower(Food.name) == name_stripped.lower()).first()
    if existing_food:
        return create_response("error", 409, f"Food with name '{name_stripped}' already exists.", existing_food.to_dict())

    image_filename_to_save = None
    if image_file:
        if image_file.filename == '':
            return create_response("error", 400, "Image file provided but has no filename.")
        # Generate a secure filename based on the food name and original extension
        extension = image_file.filename.rsplit('.', 1)[-1].lower()
        temp_filename = secure_filename(f"{name_stripped}.{extension}")

        image_save_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        try:
            image_file.save(image_save_path) # <<< --- ADD THIS LINE TO SAVE THE FILE
            image_filename_to_save = temp_filename # The actual filename saved
        except Exception as e:
            # Log this error
            print(f"Error saving image: {e}")
            return create_response("error", 500, "Could not save image file.")
            
    else: # Handle case where no image is provided, if allowed by your logic
        # If image is optional, you might set a default image_url or leave it null
        # For now, assuming image is required based on your Flutter code.
        # If not required, you might do: image_db_path = "images/default.jpg" or None
        print("No image file provided, but it is required for this endpoint.")
        pass

    # Initialize with None or empty structures
    parsed_tags = None
    parsed_contents = None
    parsed_recipe = None

    try:
        if tags_str:
            parsed_tags = json.loads(tags_str) # Parse the string into a Python list
        if contents_str:
            parsed_contents = json.loads(contents_str) # Parse into a Python list
        if recipe_str:
            parsed_recipe = json.loads(recipe_str) # Parse into a Python dict
    except json.JSONDecodeError as e:
        return create_response("error", 400, f"Invalid JSON format for tags, contents, or recipe: {e}")

    # in contents # if parsed_contents is not None find the # ingredients in the database and create a list of Ingredient objects
    try:
        # Create a new Food object
        new_food = Food(
            name=name_stripped,
            description=description,
            calories_per_serving=calories_per_serving,
            protein_per_serving=protein_per_serving,
            carbs_per_serving=carbs_per_serving,
            fat_per_serving=fat_per_serving,
            sugar_per_serving=sugar_per_serving,
            sodium_per_serving=sodium_per_serving,
            serving_size=serving_size,
            category=category,
            meal_type=meal_type,
            tags=parsed_tags,
            contents=parsed_contents,
            recipe=parsed_recipe,
            image_url=f"images/{image_filename_to_save}" if image_filename_to_save else None,
            popularity_score=popularity_score
        )
        db.session.add(new_food)  # Add the new food to the session
        # Process FoodItemIngredients based on parsed_contents
        unknown_ingredients_for_fii_add = []
        if parsed_contents and isinstance(parsed_contents, list):
            for item_content in parsed_contents:
                if isinstance(item_content, dict):
                    ingredient_name = item_content.get('ingredient')
                    quantity = item_content.get('quantity') # This will go into quantity_description
                    note = item_content.get('note')

                    if ingredient_name:
                        ingredient_obj = Ingredient.query.filter(func.lower(Ingredient.name) == ingredient_name.lower()).first()
                        if ingredient_obj:
                            fii = FoodItemIngredient(
                                ingredient=ingredient_obj, # Link to Ingredient object
                                quantity_description=quantity,
                                notes=note
                            )
                            new_food.food_ingredients_association.append(fii) # Append to the relationship
                        else:
                            unknown_ingredients_for_fii_add.append(ingredient_name)
                            app.logger.warning(f"Add Food: Ingredient '{ingredient_name}' not found. Not creating FII link for food '{name_stripped}'.")
        
        db.session.commit()  # Commit the transaction to save the food
        response_details = new_food.to_dict()
        if unknown_ingredients_for_fii_add:
            response_details['warnings_ingredients'] = f"The following ingredients provided in 'contents' were not found in the database and not linked: {', '.join(unknown_ingredients_for_fii_add)}"
        return create_response("success", 201, "Food added successfully", response_details)
    
    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return create_response("error", 500, "An error occurred", str(e))

@app.route('/allergens', methods=['GET'])
def get_all_allergens():
    try:
        allergens = Allergen.query.order_by(Allergen.name).all()
        # Using the to_dict method to get more details, or just names if preferred
        allergen_details = [allergen.to_dict() for allergen in allergens]
        # If you only want names:
        # allergen_names = [allergen.name for allergen in allergens]
        return create_response("success", 200, "Allergens retrieved successfully", allergen_details)
    except Exception as e:
        return create_response("error", 500, "An error occurred while retrieving allergens", str(e))

@app.route('/foodDetails/<int:food_id>', methods=['GET'])
def get_food_details_by_id(food_id):
    try:
        food = db.session.get(Food, food_id)
        if not food:
            return create_response("error", 404, f"Food with ID {food_id} not found.")
        
        return create_response(
            "success",
            200,
            f"Details for food '{food.name}' retrieved successfully.",
            food.to_dict() # Using the comprehensive to_dict() method
        )
    except Exception as e:
        app.logger.error(f"Error retrieving food details for ID {food_id}: {str(e)}", exc_info=True)
        return create_response("error", 500, "An error occurred while retrieving food details.", str(e))

@app.route('/foods/<int:food_id>', methods=['PUT'])
def update_food(food_id):
    food_to_update = db.session.get(Food, food_id)
    if not food_to_update:
        return create_response("error", 404, f"Food with ID {food_id} not found.")

    data = request.form
    image_file = request.files.get('image')
    updates_made = False
    old_image_filename = food_to_update.image_url
    new_image_filename_to_set = old_image_filename # Start with the current one

    try:
        # Handle name update (must be unique)
        if 'name' in data and data.get('name').strip() != food_to_update.name:
            new_name = data.get('name').strip().lower()
            if not new_name:
                return create_response("error", 400, "Food name cannot be empty.")
            existing_food_with_new_name = Food.query.filter(db.func.lower(Food.name) == new_name.lower(), Food.foodId != food_id).first()
            if existing_food_with_new_name:
                return create_response("error", 409, f"Another food with name '{new_name}' already exists.")
            food_to_update.name = new_name
            updates_made = True

        # Update other fields if present in form data
        for field in ['description', 'category', 'meal_type']:
            if field in data: # Check if key exists in form data
                # If data.get(field) is None (e.g. empty string sent for an optional field),
                # we might want to keep the old value or set it to None/empty based on model.
                # For simplicity here, if key is present, we update.
                setattr(food_to_update, field, data.get(field))
                updates_made = True

        # Handle JSON string fields: tags, contents, recipe
        for field_json_str in ['tags', 'recipe']:
            if field_json_str in data:
                json_str_value = data.get(field_json_str)
                if json_str_value: # If not empty string
                    try:
                        parsed_value = json.loads(json_str_value)
                        setattr(food_to_update, field_json_str, parsed_value)
                    except json.JSONDecodeError as e:
                        return create_response("error", 400, f"Invalid JSON format for {field_json_str}: {e}")
                else: # If empty string is provided, set to None or empty list/dict as appropriate
                    if field_json_str == 'tags' or field_json_str == 'contents':
                        setattr(food_to_update, field_json_str, []) # Default to empty list
                    elif field_json_str == 'recipe':
                        setattr(food_to_update, field_json_str, {}) # Default to empty dict
                    # If you prefer to set to None, use: setattr(food_to_update, field_json_str, None)
                updates_made = True

        # Handle 'contents' for both Food.contents and FoodItemIngredient
        unknown_ingredients_for_fii_update = []
        if 'contents' in data:
            json_str_value_contents = data.get('contents')
            parsed_contents_list = None # This will be the list of dicts for Food.contents

            if json_str_value_contents: # If not empty string
                try:
                    parsed_contents_list = json.loads(json_str_value_contents)
                    if not isinstance(parsed_contents_list, list) or \
                       not all(isinstance(item, dict) for item in parsed_contents_list):
                        raise json.JSONDecodeError("Contents must be a JSON array of ingredient objects.", json_str_value_contents, 0)
                except json.JSONDecodeError as e:
                    return create_response("error", 400, f"Invalid JSON format for contents: {e}")
            else: # Empty string for contents means clear existing and set Food.contents to empty list
                parsed_contents_list = []

            if parsed_contents_list is not None: # If 'contents' key was present
                # Delete existing FoodItemIngredient associations for this food
                FoodItemIngredient.query.filter_by(foodId=food_id).delete()

                # Add new FoodItemIngredient associations from parsed_contents_list
                for item_content in parsed_contents_list: # Iterate if list is not empty
                    ingredient_name = item_content.get('ingredient')
                    quantity = item_content.get('quantity') # maps to quantity_description
                    note = item_content.get('note')

                    if ingredient_name:
                        ingredient_obj = Ingredient.query.filter(func.lower(Ingredient.name) == ingredient_name.lower()).first()
                        if ingredient_obj:
                            fii = FoodItemIngredient(foodId=food_id, ingredientId=ingredient_obj.ingredient_id, quantity_description=quantity, notes=note)
                            db.session.add(fii)
                        else:
                            unknown_ingredients_for_fii_update.append(ingredient_name)
                            app.logger.warning(f"Update Food: Ingredient '{ingredient_name}' not found. Not creating FII link for food ID '{food_id}'.")
                
                food_to_update.contents = parsed_contents_list # Update Food.contents with the new list of dicts
                updates_made = True
                
        for field_float in ['calories_per_serving', 'protein_per_serving', 'carbs_per_serving', 'fat_per_serving', 'sugar_per_serving', 'sodium_per_serving', 'serving_size', 'popularity_score']:
            if field_float in data:
                try:
                    value_str = data.get(field_float)
                    if value_str is not None and value_str.strip() != "": # Ensure it's not empty string before float conversion
                        value = float(value_str)
                        setattr(food_to_update, field_float, value)
                        updates_made = True
                except (ValueError, TypeError):
                    return create_response("error", 400, f"Invalid value for {field_float}. Must be a number.")

        # Handle image update
        current_food_name_for_filename = food_to_update.name # Use the potentially updated name

        if image_file: # New image uploaded
            if image_file.filename == '':
                return create_response("error", 400, "Image file provided but has no filename.")
            extension = image_file.filename.rsplit('.', 1)[-1].lower()
            new_image_filename_to_set = secure_filename(f"{current_food_name_for_filename}.{extension}")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], new_image_filename_to_set)
            image_file.save(save_path) # Save the new image

            # If old image exists and is different, delete it
            if old_image_filename and old_image_filename != new_image_filename_to_set:
                old_file_path = os.path.join(app.config['UPLOAD_FOLDER'], old_image_filename)
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)
            updates_made = True
        elif 'name' in data and old_image_filename and data.get('name').strip() != old_image_filename.rsplit('.',1)[0]: # Name changed, no new image, but old image exists
            # If food name changed, and an old image exists, rename the old image
            old_ext = old_image_filename.rsplit('.', 1)[-1] if '.' in old_image_filename else ''
            new_filename_after_rename = secure_filename(f"{current_food_name_for_filename}.{old_ext}")
            if old_image_filename != new_filename_after_rename: # Ensure rename is necessary
                old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_image_filename)
                new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename_after_rename)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                new_image_filename_to_set = new_filename_after_rename
            updates_made = True

        if updates_made:
            food_to_update.image_url = "images/" + new_image_filename_to_set
            food_to_update.updated_at = datetime.utcnow()
            db.session.commit()
            return create_response("success", 200, "Food updated successfully.", food_to_update.to_dict())
        else:
            return create_response("info", 200, "No changes detected or no data provided for update.", food_to_update.to_dict())

    except IntegrityError: # Handles potential unique constraint violation on name if logic above missed it
        db.session.rollback()
        return create_response("error", 409, f"Food name '{food_to_update.name}' likely already exists for another item.")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error updating food ID {food_id}: {str(e)}", exc_info=True)
        return create_response("error", 500, "An error occurred while updating the food.", str(e))


@app.route('/allergens', methods=['POST'])
def add_allergen():
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    data = request.get_json()
    name = data.get('name')
    description = data.get('description', None) # Optional description
    ingredient_names_raw = data.get('ingredients', []) # Expects a list of ingredient names

    if not name or not name.strip():
        return create_response("error", 400, "Allergen name is required")

    allergen_name_stripped = name.strip()
    allergen_name_lower = allergen_name_stripped.lower()

    # Case-insensitive check for existing allergen
    existing_allergen = Allergen.query.filter(db.func.lower(Allergen.name) == allergen_name_lower).first()
    if existing_allergen:
        return create_response("error", 409, f"Allergen '{allergen_name_stripped}' already exists.", existing_allergen.to_dict())

    try:
        new_allergen = Allergen(name=allergen_name_stripped, description=description)
        db.session.add(new_allergen)

        processed_ingredient_names = normalize_string_list(ingredient_names_raw)
        associated_ingredients_count = 0
        unknown_ingredients_input = []

        if processed_ingredient_names:
            for ingredient_name_norm in processed_ingredient_names:
                # Case-insensitive check for existing ingredient
                ingredient_obj = Ingredient.query.filter(db.func.lower(Ingredient.name) == ingredient_name_norm).first()
                if ingredient_obj:
                    new_allergen.ingredients.append(ingredient_obj) # SQLAlchemy handles the association table
                    associated_ingredients_count += 1
                else:
                    # Find the original casing for the unknown ingredient name for reporting
                    original_unknown_name = next((raw_name for raw_name in ingredient_names_raw if isinstance(raw_name, str) and raw_name.lower().strip() == ingredient_name_norm), ingredient_name_norm)
                    unknown_ingredients_input.append(original_unknown_name)
        
        db.session.commit()

        response_message = f"Allergen '{new_allergen.name}' added successfully."
        details = new_allergen.to_dict()
        if unknown_ingredients_input:
            response_message += f" {associated_ingredients_count} ingredient(s) associated."
            details['warnings'] = {
                "unknown_ingredients_provided": unknown_ingredients_input,
                "note": "These ingredient names were provided but not found in the database and were not associated."
            }
        return create_response("success", 201, response_message, details)

    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while adding the allergen", str(e))

@app.route('/allergens/<int:allergen_id>', methods=['PUT'])
def update_allergen(allergen_id):
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    data = request.get_json()

    try:
        # Find the allergen by ID
        allergen = db.session.get(Allergen, allergen_id)
        if not allergen:
            return create_response("error", 404, f"Allergen with ID {allergen_id} not found")

        updates_made = False
        details = allergen.to_dict() # Start with current details
        unknown_ingredients_input = []

        # Update description if provided and not empty/null
        if 'description' in data:
            new_description = data.get('description')
            if isinstance(new_description, str) and new_description.strip():
                allergen.description = new_description.strip()
                updates_made = True
            # If provided but empty/null, we explicitly don't update, keeping the old value

        # Update associated ingredients if provided and the list is not empty after normalization
        if 'ingredients' in data:
            ingredient_names_raw = data.get('ingredients', [])
            processed_ingredient_names = normalize_string_list(ingredient_names_raw)

            if processed_ingredient_names:
                # Clear existing associations
                allergen.ingredients.clear()
                associated_ingredients_count = 0

                # Find and associate new ingredients
                for ingredient_name_norm in processed_ingredient_names:
                    ingredient_obj = Ingredient.query.filter(db.func.lower(Ingredient.name) == ingredient_name_norm).first()
                    if ingredient_obj:
                        allergen.ingredients.append(ingredient_obj)
                        associated_ingredients_count += 1
                    else:
                        original_unknown_name = next((raw_name for raw_name in ingredient_names_raw if isinstance(raw_name, str) and raw_name.lower().strip() == ingredient_name_norm), ingredient_name_norm)
                        unknown_ingredients_input.append(original_unknown_name)

                updates_made = True
                details = allergen.to_dict() # Refresh details to include new associations

        if updates_made:
            db.session.commit()
            response_message = f"Allergen '{allergen.name}' updated successfully."
            if unknown_ingredients_input:
                 details['warnings'] = {
                    "unknown_ingredients_provided": unknown_ingredients_input,
                    "note": "These ingredient names were provided but not found in the database and were not associated."
                }
            return create_response("success", 200, response_message, details)
        else:
            return create_response("error", 400, "No valid fields provided for update or provided values were empty/null")

    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while updating the allergen", str(e))

@app.route('/ingredients/', defaults={'ingredient_name': None}, methods=['GET'])
@app.route('/ingredients/<path:ingredient_name>', methods=['GET'])
def get_ingredients_by_name(ingredient_name):
    try:
        if not ingredient_name:
            ingredients = Ingredient.query.order_by(Ingredient.name).all()
            ingredient_names = [ingredient.to_dict() for ingredient in ingredients]
            return create_response(
                "success",
                200,
                f"Retrieved all {len(ingredients)} ingredients",
                ingredient_names
            )

        search_term = ingredient_name.lower()
        ingredients = Ingredient.query.filter(Ingredient.name.ilike(f'%{search_term}%')).order_by(Ingredient.name).all()

        if not ingredients:
            return create_response("error", 404, f"No ingredients found matching '{ingredient_name}'")

        ingredient_names = [ingredient.to_dict() for ingredient in ingredients]
        return create_response(
            "success",
            200,
            f"Found {len(ingredients)} ingredient(s) matching '{ingredient_name}'",
            ingredient_names
        )
    except Exception as e:
        return create_response("error", 500, "An error occurred while retrieving ingredients", str(e))

@app.route('/allergens/', defaults={'allergen_name': None}, methods=['GET'])
@app.route('/allergens/<path:allergen_name>', methods=['GET'])
def get_allergens_by_name(allergen_name):
    try:
        if not allergen_name:
            allergens = Allergen.query.order_by(Allergen.name).all()
            allergen_names = [allergen.to_dict() for allergen in allergens]
            return create_response("success", 200, f"Retrieved all {len(allergens)} allergens", allergen_names)

        search_term = allergen_name.lower()
        allergens = Allergen.query.filter(Allergen.name.ilike(f'%{search_term}%')).order_by(Allergen.name).all()

        if not allergens:
            return create_response("error", 404, f"No allergens found matching '{allergen_name}'")

        allergen_names = [allergen.to_dict() for allergen in allergens]
        return create_response("success", 200, f"Found {len(allergens)} allergen(s) matching '{allergen_name}'", allergen_names)
    except Exception as e:
        return create_response("error", 500, "An error occurred while retrieving allergens", str(e))

@app.route('/ingredients', methods=['GET'])
def get_all_ingredients():
    try:
        ingredients = Ingredient.query.order_by(Ingredient.name).all()
        ingredient_names = [ingredient.to_dict() for ingredient in ingredients]
        return create_response("success", 200, "Ingredients retrieved successfully", ingredient_names)
    except Exception as e:
        return create_response("error", 500, "An error occurred while retrieving ingredients", str(e))

@app.route('/ingredients', methods=['POST'])
def add_ingredient():
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    data = request.get_json()
    name = data.get('name')
    allergen_names_raw = data.get('allergens', []) # Expects a list of allergen names

    if not name or not name.strip():
        return create_response("error", 400, "Ingredient name is required")

    ingredient_name_stripped = name.strip()
    ingredient_name_lower = ingredient_name_stripped.lower()

    # Case-insensitive check for existing ingredient
    existing_ingredient = Ingredient.query.filter(db.func.lower(Ingredient.name) == ingredient_name_lower).first()
    if existing_ingredient:
        return create_response("error", 409, f"Ingredient '{ingredient_name_stripped}' already exists.", existing_ingredient.to_dict())

    try:
        new_ingredient = Ingredient(name=ingredient_name_stripped)
        db.session.add(new_ingredient)

        processed_allergen_names = normalize_string_list(allergen_names_raw)
        associated_allergens_count = 0
        unknown_allergens_input = []

        if processed_allergen_names:
            for allergen_name_norm in processed_allergen_names:
                # Case-insensitive check for existing allergen
                allergen_obj = Allergen.query.filter(db.func.lower(Allergen.name) == allergen_name_norm).first()
                if allergen_obj:
                    new_ingredient.allergens.append(allergen_obj)
                    associated_allergens_count += 1
                else:
                    # Find the original casing for the unknown allergen name for reporting
                    original_unknown_name = next((raw_name for raw_name in allergen_names_raw if isinstance(raw_name, str) and raw_name.lower().strip() == allergen_name_norm), allergen_name_norm)
                    unknown_allergens_input.append(original_unknown_name)
        
        db.session.commit()

        response_message = f"Ingredient '{new_ingredient.name}' added successfully."
        details = new_ingredient.to_dict()
        if unknown_allergens_input:
            response_message += f" {associated_allergens_count} allergen(s) associated."
            details['warnings'] = {
                "unknown_allergens_provided": unknown_allergens_input,
                "note": "These allergen names were provided but not found in the database and were not associated."
            }
        return create_response("success", 201, response_message, details)

    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while adding the ingredient", str(e))

@app.route('/ingredients/<int:ingredient_id>', methods=['PUT'])
def update_ingredient(ingredient_id):
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    data = request.get_json()

    try:
        # Find the ingredient by ID
        ingredient = db.session.get(Ingredient, ingredient_id)
        if not ingredient:
            return create_response("error", 404, f"Ingredient with ID {ingredient_id} not found")

        updates_made = False
        details = ingredient.to_dict() # Start with current details
        unknown_allergens_input = []

        # Update associated allergens if provided and the list is not empty after normalization
        if 'allergens' in data:
            allergen_names_raw = data.get('allergens', [])
            processed_allergen_names = normalize_string_list(allergen_names_raw)

            if processed_allergen_names:
                # Clear existing associations
                ingredient.allergens.clear()
                associated_allergens_count = 0

                # Find and associate new allergens
                for allergen_name_norm in processed_allergen_names:
                    allergen_obj = Allergen.query.filter(db.func.lower(Allergen.name) == allergen_name_norm).first()
                    if allergen_obj:
                        ingredient.allergens.append(allergen_obj)
                        associated_allergens_count += 1
                    else:
                        original_unknown_name = next((raw_name for raw_name in allergen_names_raw if isinstance(raw_name, str) and raw_name.lower().strip() == allergen_name_norm), allergen_name_norm)
                        unknown_allergens_input.append(original_unknown_name)

                updates_made = True
                details = ingredient.to_dict() # Refresh details to include new associations
            else:
                ingredient.allergens.clear()
                updates_made = True
                details = ingredient.to_dict()

        if updates_made:
            db.session.commit()
            response_message = f"Ingredient '{ingredient.name}' updated successfully."
            if unknown_allergens_input:
                 details['warnings'] = {
                    "unknown_allergens_provided": unknown_allergens_input,
                    "note": "These allergen names were provided but not found in the database and were not associated."
                }
            return create_response("success", 200, response_message, details)
        else:
            return create_response("error", 400, "No valid fields provided for update or provided values were empty/null")
    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while updating the ingredient", str(e))


@app.route('/medical_condition_guidelines', methods=['POST'])
def add_medical_condition_guideline():
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    data = request.get_json()

    condition_name = data.get('condition_name')
    guideline_type = data.get('guideline_type')
    parameter_target = data.get('parameter_target')
    parameter_target_type = data.get('parameter_target_type')
    # Optional fields
    threshold_value = data.get('threshold_value')
    threshold_unit = data.get('threshold_unit')
    severity = data.get('severity')
    description = data.get('description')

    if not all([condition_name, guideline_type, parameter_target, parameter_target_type]):
        return create_response("error", 400, "Missing required fields: condition_name, guideline_type, parameter_target, parameter_target_type")

    try:
        new_guideline = MedicalConditionDietaryGuideline(
            condition_name=condition_name.strip().lower(),  # Normalize condition name to lowercase
            guideline_type=guideline_type,
            parameter_target=parameter_target,
            parameter_target_type=parameter_target_type,
            threshold_value=threshold_value,
            threshold_unit=threshold_unit,
            severity=severity,
            description=description
        )
        db.session.add(new_guideline)
        db.session.commit()
        return create_response("success", 201, "Medical condition dietary guideline added successfully", new_guideline.to_dict())
    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while adding the guideline", str(e))

@app.route('/update_add_in_bulk/medical_condition_guidelines', methods=['POST'])
def bulk_update_medical_condition_guidelines():
    if not request.is_json:
        return create_response("error", 415, "Content-Type must be application/json")

    data = request.get_json()
    guideline_details_list = data.get('details')

    if not isinstance(guideline_details_list, list):
        return create_response("error", 400, "Request body must contain a 'details' list.")

    results = {
        "created": [],
        "updated": [],
        "deleted": [],
        "errors": []
    }

    try:
        for item_index, item_data in enumerate(guideline_details_list):
            guideline_id = item_data.get('guideline_id')
            delete_flag = item_data.get('delete', False)

            if delete_flag:
                if guideline_id:
                    guideline_to_delete = db.session.get(MedicalConditionDietaryGuideline, guideline_id)
                    if guideline_to_delete:
                        db.session.delete(guideline_to_delete)
                        results["deleted"].append({"guideline_id": guideline_id, "status": "deleted"})
                    else:
                        results["errors"].append({"guideline_id": guideline_id, "status": "delete_failed", "reason": "not_found"})
                # If delete is true and no guideline_id, do nothing as per requirement
            else: # Create or Update
                condition_name = item_data.get('condition_name')
                guideline_type = item_data.get('guideline_type')
                parameter_target = item_data.get('parameter_target')
                parameter_target_type = item_data.get('parameter_target_type')

                if not all([condition_name, guideline_type, parameter_target, parameter_target_type]):
                    results["errors"].append({"index": item_index, "status": "validation_failed", "reason": "Missing required fields for create/update"})
                    continue

                if guideline_id: # Update existing
                    guideline_to_update = db.session.get(MedicalConditionDietaryGuideline, guideline_id)
                    if guideline_to_update:
                        guideline_to_update.condition_name = condition_name.strip().lower()
                        guideline_to_update.guideline_type = guideline_type
                        guideline_to_update.parameter_target = parameter_target
                        guideline_to_update.parameter_target_type = parameter_target_type
                        guideline_to_update.threshold_value = item_data.get('threshold_value')
                        guideline_to_update.threshold_unit = item_data.get('threshold_unit')
                        guideline_to_update.severity = item_data.get('severity')
                        guideline_to_update.description = item_data.get('description')
                        guideline_to_update.updated_at = datetime.utcnow()
                        results["updated"].append({"guideline_id": guideline_id, "status": "updated"})
                    else:
                        results["errors"].append({"guideline_id": guideline_id, "status": "update_failed", "reason": "not_found"})
                else: # Create new
                    new_guideline = MedicalConditionDietaryGuideline(
                        condition_name=condition_name.strip().lower(),
                        guideline_type=guideline_type,
                        parameter_target=parameter_target,
                        parameter_target_type=parameter_target_type,
                        threshold_value=item_data.get('threshold_value'),
                        threshold_unit=item_data.get('threshold_unit'),
                        severity=item_data.get('severity'),
                        description=item_data.get('description')
                    )
                    db.session.add(new_guideline)
                    # We can't get the ID until after flush/commit, so we'll report success generically
                    results["created"].append({"index": item_index, "status": "marked_for_creation"})
        
        db.session.commit()
        # For created items, now that they are committed, you could re-fetch them if you need their IDs in the response.
        # For simplicity, this example just reports they were marked for creation.

        return create_response("success", 200, "Bulk operation processed.", results)

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in bulk_update_medical_condition_guidelines: {str(e)}", exc_info=True)
        results["errors"].append({"status": "transaction_failed", "reason": str(e)})
        return create_response("error", 500, "An error occurred during bulk operation.", results)


@app.route('/medical_condition_guidelines/', defaults={'condition_name': None}, methods=['GET'])
@app.route('/medical_condition_guidelines/<path:condition_name>', methods=['GET'])
def get_medical_condition_guidelines(condition_name): # Renamed for clarity
    try:
        grouped_conditions = {} # Use a dictionary to group by condition name
        message_template = ""

        if not condition_name:
            # Case 1: Get all medical conditions and their guidelines
            all_guidelines_query = db.session.query(
                MedicalConditionDietaryGuideline.condition_name,
                MedicalConditionDietaryGuideline.description
            ).distinct().order_by(MedicalConditionDietaryGuideline.condition_name).all()

            if not all_guidelines_query:
                return create_response("success", 200, "No medical condition guidelines found.", [])

            for c_name, desc in all_guidelines_query:
                # Names are stored in lowercase, display them in Title Case
                display_name = c_name.title() 
                if display_name not in grouped_conditions:
                    grouped_conditions[display_name] = {
                        "name": display_name,
                        "descriptions": []
                    }
                # Add description if it's not None and not already present for this condition
                if desc and desc not in grouped_conditions[display_name]["descriptions"]:
                    grouped_conditions[display_name]["descriptions"].append(desc)
            
            message_template = "Retrieved {count} unique medical condition(s) with their guidelines"

        else:
            # Case 2: Search for medical conditions by name
            search_term_normalized = condition_name.lower()
            
            matching_guidelines_query = db.session.query(
                MedicalConditionDietaryGuideline.condition_name,
                MedicalConditionDietaryGuideline.description
            ).filter(
                db.func.lower(MedicalConditionDietaryGuideline.condition_name).ilike(f'%{search_term_normalized}%')
            ).order_by(MedicalConditionDietaryGuideline.condition_name).all()

            if not matching_guidelines_query:
                return create_response(
                    "success", 
                    200, 
                    f"No medical conditions found matching '{condition_name}'", 
                    []
                )

            for c_name, desc in matching_guidelines_query:
                display_name = c_name.title() # Names are stored in lowercase
                if display_name not in grouped_conditions:
                    grouped_conditions[display_name] = {
                        "name": display_name,
                        "descriptions": []
                    }
                if desc and desc not in grouped_conditions[display_name]["descriptions"]:
                     grouped_conditions[display_name]["descriptions"].append(desc)
            
            message_template = "Found {count} medical condition(s) matching '{search_term}'"

        response_details = list(grouped_conditions.values())
        found_count = len(response_details)
        
        if not condition_name:
            message = message_template.format(count=found_count)
        else:
            message = message_template.format(count=found_count, search_term=condition_name)

        return create_response(
            "success",
            200,
            message,
            response_details
        )

    except Exception as e:
        app.logger.error(f"Error in get_medical_condition_guidelines: {str(e)}", exc_info=True)
        return create_response("error", 500, "An error occurred while retrieving medical condition names", str(e))

@app.route('/medical_condition_guidelines/condition/<path:condition_name>', methods=['GET'])
def get_guidelines_by_condition_name(condition_name):
    try:
        # Perform a case-insensitive exact match for condition_name
        # If you need a partial match, you can use .ilike(f'%{condition_name}%')
        guidelines = MedicalConditionDietaryGuideline.query.filter(
            db.func.lower(MedicalConditionDietaryGuideline.condition_name) == condition_name.lower()
        ).order_by(MedicalConditionDietaryGuideline.guideline_type).all()

        if not guidelines:
            return create_response("success", 200, f"No guidelines found for condition: {condition_name}", [])

        return create_response(
            "success",
            200,
            f"Guidelines for condition '{condition_name}' retrieved successfully",
            [g.to_dict() for g in guidelines]
        )
    except Exception as e:
        return create_response("error", 500, "An error occurred while retrieving guidelines by condition name", str(e))


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
                return create_response("error", 400, f"Invalid meal type. Must be one of {valid_meal_types}",None)

        # Find the food by name
        food = Food.query.filter(Food.name.ilike(food_name.lower())).first()
        if not food:
            return create_response("error", 204, f"Food '{food_name}' not found in database",None)
        
        # Check for duplicate entries within last 5 minutes
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        recent_scan = ScannedHistory.query.filter(
            ScannedHistory.userId == userid,
            ScannedHistory.foodId == food.foodId,
            ScannedHistory.scanned_at >= five_minutes_ago
        ).first()

        if recent_scan:
            streak_data = calculate_streak(userid)
            return create_response(
                "error", 
                203, 
                "Duplicate scan detected. Please wait 5 minutes before scanning the same food again.",
                streak_data
            )
        
        
        user = db.session.get(User, userid)
        if not user:
            return create_response("error", 404, "User not found")

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

        # Get scan data and streak information
        scan_dict = new_scan.to_dict()
        streak_data = calculate_streak(userid)
        update_challenge_progress(food.foodId, meal_type, serving_size, userid)
        
        # Combine the data
        response_data = {**scan_dict, **streak_data}

        return create_response(
            "success", 
            201, 
            "Food scan recorded successfully",
            response_data
        )

    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while recording food scan", str(e))


@app.route('/home/<path:userid>', methods=['GET'])
def get_home_data(userid):
    try:
        # Get user and validate
        user = db.session.get(User, userid)
        if not user:
            return create_response("error", 204, "User not found")

        # Calculate streak data
        streak_data = calculate_streak(userid)
        
        # You can add more home data here if needed
        response_data = {
            'current_streak': streak_data['current_streak'],
            'longest_streak': streak_data['longest_streak']
        }

        return create_response(
            "success",
            200,
            "Home data retrieved successfully",
            response_data
        )

    except Exception as e:
        return create_response("error", 500, "An error occurred while retrieving home data", str(e))


@app.route('/profile/<int:userid>', methods=['GET'])
def get_user_profile(userid):
    try:
        profile_data = get_profile(userid)
        if profile_data is None:
            return create_response("error", 404, "User not found")
            
        return create_response(
            "success",
            200,
            "Profile data retrieved successfully",
            profile_data
        )
    except Exception as e:
        return create_response("error", 500, "Error retrieving profile data", str(e))


@app.route('/update_profile/<int:userid>', methods=['PUT'])
def update_user_profile(userid):
    try:
        # Check if request is JSON
        if not request.is_json:
            return create_response("error", 415, "Content-Type must be application/json")

        # Get user
        user = db.session.get(User, userid)
        if not user:
            return create_response("error", 404, "User not found")

        # Get JSON data
        data = request.get_json()
        updates_made = []

        # Update email if provided
        if 'account_email' in data:
            user.email = data['account_email']
            updates_made.append('email')

        # Update username if provided
        if 'account_username' in data:
            user.username = data['account_username']
            updates_made.append('username')

        # Update password if provided
        if 'password' in data:
            user.password = data['password']
            updates_made.append('password')

        # Update activity level if provided
        if 'health_activity_level' in data:
            activity_level_map = {'Not Active': -1, 'Moderate': 0, 'Very Active': 1}
            activity_level = activity_level_map.get(data['health_activity_level'])
            if activity_level is not None:
                user.activity_level = activity_level
                updates_made.append('activity_level')
            else:
                return create_response("error", 400, "Invalid activity level value")

        # Update goal if provided
        if 'health_goal' in data:
            goal_map = {'Go Slim': -1, 'Maintain': 0, 'Gain Weight': 1}
            goal = goal_map.get(data['health_goal'])
            if goal is not None:
                user.goal = goal
                updates_made.append('goal')
            else:
                return create_response("error", 400, "Invalid goal value")

        # Update dietary preference if provided
        if 'meal_dietary_pref' in data:
            user.dietary_pref = data['meal_dietary_pref']
            updates_made.append('dietary_preference')

        # Update meal times if provided
        meal_times_updated = False
        current_meal_times = user.meal_times or {}

        if 'meal_breakfast_time' in data:
            current_meal_times['breakfast'] = data['meal_breakfast_time']
            meal_times_updated = True

        if 'meal_lunch_time' in data:
            current_meal_times['lunch'] = data['meal_lunch_time']
            meal_times_updated = True

        if 'meal_dinner_time' in data:
            current_meal_times['dinner'] = data['meal_dinner_time']
            meal_times_updated = True

        if meal_times_updated:
            user.meal_times = current_meal_times
            updates_made.append('meal_times')

        # Commit changes if any updates were made
        if updates_made:
            try:
                db.session.commit()
                return create_response(
                    "success",
                    200,
                    "Profile updated successfully",
                    {
                        "user": user.to_dict()
                    }
                )
            except IntegrityError:
                db.session.rollback()
                return create_response("error", 409, "Username or email already exists")
        else:
            return create_response("error", 400, "No valid fields provided for update")

    except Exception as e:
        db.session.rollback()
        return create_response("error", 500, "An error occurred while updating profile", str(e))


@app.route('/leaderboard/season/<int:userid>', methods=['GET'])
def season_leaderboard(userid):
    try:
        rankings = get_this_season_leaderboards(userid)
        return create_response("success", 200, "Season leaderboard retrieved successfully", rankings)
    except Exception as e:
        return create_response("error", 500, "Error retrieving season leaderboard", str(e))

@app.route('/leaderboard/week/<int:userid>', methods=['GET'])
def week_leaderboard(userid):
    try:
        rankings = get_this_week_leaderboards(userid)
        return create_response("success", 200, "Weekly leaderboard retrieved successfully", rankings)
    except Exception as e:
        return create_response("error", 500, "Error retrieving weekly leaderboard", str(e))

@app.route('/leaderboard/month/<int:userid>', methods=['GET'])
def month_leaderboard(userid):
    try:
        rankings = get_this_month_leaderboards(userid)
        return create_response("success", 200, "Monthly leaderboard retrieved successfully", rankings)
    except Exception as e:
        return create_response("error", 500, "Error retrieving monthly leaderboard", str(e))

@app.route('/scanned_details/<int:userid>/<int:months_back>', methods=['GET'])
def get_user_scanned_details(userid, months_back):
    try:
        scanned_details = get_scanned_details(userid, months_back)
        if not scanned_details:
            return create_response("success", 200, "No scanned foods found for the specified month", [])
            
        return create_response(
            "success",
            200,
            f"Scanned food details retrieved successfully for {months_back} months ago",
            scanned_details
        )
    except Exception as e:
        return create_response("error", 500, "Error retrieving scanned food details", str(e))


@app.route('/nutrients/', defaults={'search_term': None}, methods=['GET'])
@app.route('/nutrients/<path:search_term>', methods=['GET'])
def get_nutrients(search_term):
    try:
        nutrient_list_names = ["Calories", "Protein", "Carbs", "Fat", "Sugar", "Sodium"]
        
        if search_term:
            search_term_lower = search_term.lower()
            filtered_nutrients = [n for n in nutrient_list_names if search_term_lower in n.lower()]
        else:
            filtered_nutrients = nutrient_list_names
            
        # Format as list of {"name": "value"}
        formatted_nutrients = [{"name": nutrient} for nutrient in filtered_nutrients]
        
        message = f"Found {len(formatted_nutrients)} nutrient(s)"
        if search_term:
            message += f" matching '{search_term}'"
        else:
            message = "Nutrients retrieved successfully"

        return create_response(
            "success",
            200,
            message,
            formatted_nutrients
        )
    except Exception as e:
        app.logger.error(f"Error in get_nutrients: {str(e)}", exc_info=True)
        return create_response("error", 500, "An error occurred while retrieving nutrients", str(e))

@app.route('/food_categories/', defaults={'search_term': None}, methods=['GET'])
@app.route('/food_categories/<path:search_term>', methods=['GET'])
def get_food_categories(search_term):
    try:
        categories_query = db.session.query(Food.category).distinct().order_by(Food.category).all()
        # Extract names, filter out None or empty strings, and ensure uniqueness again
        category_names_raw = sorted(list(set(c[0] for c in categories_query if c[0] and c[0].strip())))
        
        if search_term:
            search_term_lower = search_term.lower()
            filtered_categories = [c for c in category_names_raw if search_term_lower in c.lower()]
        else:
            filtered_categories = category_names_raw
            
        formatted_categories = [{"name": category} for category in filtered_categories]
        
        message = f"Found {len(formatted_categories)} food category(s)"
        if search_term:
            message += f" matching '{search_term}'"
        else:
            message = "Food categories retrieved successfully"

        return create_response(
            "success",
            200,
            message,
            formatted_categories
        )
    except Exception as e:
        app.logger.error(f"Error in get_food_categories: {str(e)}", exc_info=True)
        return create_response("error", 500, "An error occurred while retrieving food categories", str(e))

@app.route('/food_tags/', defaults={'search_term': None}, methods=['GET'])
@app.route('/food_tags/<path:search_term>', methods=['GET'])
def get_food_tags(search_term):
    try:
        # Static list of meaningful food tags
        food_tag_list_names = [
            "vegetarian", "vegan", "gluten-free", "dairy-free", "nut-free",
            "low-carb", "high-protein", "low-fat", "low-sugar", "low-sodium",
            "quick-meal", "easy-to-make", "comfort-food", "healthy", "organic",
            "spicy", "sweet", "sour", "savory", "breakfast-item", "lunch-item",
            "dinner-item", "snack", "dessert", "baking", "grilling", "slow-cooker","chicken","crispy","fried","fast-food",
            "indian","nepali","chinese","italian","mexican","japanese","korean"
        ]
        
        if search_term:
            search_term_lower = search_term.lower()
            filtered_tags = [t for t in food_tag_list_names if search_term_lower in t.lower()]
        else:
            filtered_tags = food_tag_list_names
            
        # Format as list of {"name": "value"}
        formatted_tags = [{"name": tag} for tag in sorted(filtered_tags)]

        message = f"Found {len(formatted_tags)} food tag(s)"
        if search_term:
            message += f" matching '{search_term}'"
        else:
            message = "Food tags retrieved successfully"

        return create_response(
            "success",
            200,
            message,
            formatted_tags
        )
    except Exception as e:
        # In a real app, you'd use app.logger.error
        app.logger.error(f"Error in get_food_tags: {str(e)}", exc_info=True)
        return create_response("error", 500, "An error occurred while retrieving food tags", str(e))

def get_scanned_details(userid, months_back):
    try:
        # Calculate start and end dates for the requested month
        today = datetime.today()
        if months_back == 0:
            # For current month
            start_date = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if today.month == 12:
                end_date = today.replace(day=31, hour=23, minute=59, second=59, microsecond=999999)
            else:
                next_month = today.replace(day=1) + relativedelta(months=1)
                end_date = (next_month - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            # For previous months
            start_date = (today.replace(day=1) - relativedelta(months=months_back)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = (start_date + relativedelta(months=1) - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)


        # Get all scans for the specified month with food details
        scans = db.session.query(
            ScannedHistory,
            Food
        ).join(
            Food, ScannedHistory.foodId == Food.foodId
        ).filter(
            ScannedHistory.userId == userid,
            db.func.date(ScannedHistory.scanned_at) >= start_date,
            db.func.date(ScannedHistory.scanned_at) <= end_date
        ).order_by(
            ScannedHistory.scanned_at.desc()
        ).all()

        # Organize scans by date
        scanned_details = {}
        for scan, food in scans:
            scan_date = scan.scanned_at.strftime('%Y-%m-%d')
            
            if scan_date not in scanned_details:
                scanned_details[scan_date] = []
            
            # Create food details dictionary
            food_details = {
                'name': food.name,
                'meal_type': scan.meal_time,
                'scanned_at': scan.scanned_at.strftime('%H:%M'),
                'servings': scan.servings,
                'calories': food.calories_per_serving * scan.servings,
                'protein': food.protein_per_serving * scan.servings,
                'carbs': food.carbs_per_serving * scan.servings,
                'fat': food.fat_per_serving * scan.servings,
                'category': food.category
            }
            
            scanned_details[scan_date].append(food_details)

        # Convert to list of dictionaries sorted by date
        formatted_response = [
            {
                'date': date,
                'food_details': details
            }
            for date, details in sorted(scanned_details.items(), reverse=True)
        ]

        return formatted_response

    except Exception as e:
        print(f"Error in get_scanned_details: {str(e)}")
        return []

def update_challenge_progress(foodid, meal_time, serving_size, userid):
    try:
        # Get current time for deadline comparison
        current_time = datetime.now()

        # Get user's accepted challenges that haven't passed deadline
        active_challenges = db.session.query(AcceptedChallenge, Challenge).join(
            Challenge,
            AcceptedChallenge.challengeId == Challenge.challengeId
        ).filter(
            AcceptedChallenge.userId == userid,
            AcceptedChallenge.completed == False,
            Challenge.deadline > current_time
        ).all()

        # Get the scanned food details using Session.get() instead of Query.get()
        food = db.session.get(Food, foodid)
        if not food:
            return

        # Calculate actual nutritional values based on serving size
        serving_ratio = serving_size / float(food.serving_size)
        actual_protein = food.protein_per_serving * serving_ratio
        actual_carbs = food.carbs_per_serving * serving_ratio
        actual_calories = food.calories_per_serving * serving_ratio
        actual_fat = food.fat_per_serving * serving_ratio
        actual_sugar = food.sugar_per_serving * serving_ratio
        actual_sodium = food.sodium_per_serving * serving_ratio
        print(actual_protein, actual_carbs, actual_calories,serving_ratio)

        # Process each active challenge
        for accepted, challenge in active_challenges:
            requirements = challenge.requirements
            if not isinstance(requirements, dict) or not requirements:
                continue # Skip if no requirements or not a dict

            progress_changed_in_this_update = False
            
            # List to store the percentage progress this scan makes towards each individual requirement's *total* target
            individual_scan_contributions_pct = []

            if 'required_protein' in requirements and requirements['required_protein'] > 0:
                protein_scan_contrib_pct = (actual_protein / requirements['required_protein']) * 100
                individual_scan_contributions_pct.append(protein_scan_contrib_pct)

            if 'required_calories' in requirements and requirements['required_calories'] > 0:
                calorie_scan_contrib_pct = (actual_calories / requirements['required_calories']) * 100
                individual_scan_contributions_pct.append(calorie_scan_contrib_pct)

            if 'required_carbs' in requirements and requirements['required_carbs'] > 0:
                carbs_scan_contrib_pct = (actual_carbs / requirements['required_carbs']) * 100
                individual_scan_contributions_pct.append(carbs_scan_contrib_pct)
            
            if 'required_fat' in requirements and requirements['required_fat'] > 0:
                fat_scan_contrib_pct = (actual_fat / requirements['required_fat']) * 100
                individual_scan_contributions_pct.append(fat_scan_contrib_pct)

            if 'required_sugar' in requirements and requirements['required_sugar'] > 0:
                sugar_scan_contrib_pct = (actual_sugar / requirements['required_sugar']) * 100
                individual_scan_contributions_pct.append(sugar_scan_contrib_pct)

            if 'required_sodium' in requirements and requirements['required_sodium'] > 0:
                sodium_scan_contrib_pct = (actual_sodium / requirements['required_sodium']) * 100
                individual_scan_contributions_pct.append(sodium_scan_contrib_pct)

            if 'min_protein_per_serving' in requirements and requirements['min_protein_per_serving'] > 0:
                if actual_protein < requirements['min_protein_per_serving']:
                    # If the scan does not meet the minimum protein requirement, skip this challenge
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to insufficient protein: {actual_protein} < {requirements['min_protein_per_serving']}")
                    continue
                else:
                    # If it meets the minimum protein requirement, we can consider it a valid contribution
                    individual_scan_contributions_pct.append(100)

            if 'min_calories_per_serving' in requirements and requirements['min_calories_per_serving'] > 0:
                if actual_calories < requirements['min_calories_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to insufficient calories: {actual_calories} < {requirements['min_calories_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'min_carbs_per_serving' in requirements and requirements['min_carbs_per_serving'] > 0:
                if actual_carbs < requirements['min_carbs_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to insufficient carbs: {actual_carbs} < {requirements['min_carbs_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'min_fat_per_serving' in requirements and requirements['min_fat_per_serving'] > 0:
                if actual_fat < requirements['min_fat_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to insufficient fat: {actual_fat} < {requirements['min_fat_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'min_sugar_per_serving' in requirements and requirements['min_sugar_per_serving'] > 0:
                if actual_sugar < requirements['min_sugar_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to insufficient sugar: {actual_sugar} < {requirements['min_sugar_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'min_sodium_per_serving' in requirements and requirements['min_sodium_per_serving'] > 0:
                if actual_sodium < requirements['min_sodium_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to insufficient sodium: {actual_sodium} < {requirements['min_sodium_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'max_protein_per_serving' in requirements and requirements['max_protein_per_serving'] > 0:
                if actual_protein > requirements['max_protein_per_serving']:
                    # If the scan exceeds the maximum protein requirement, skip this challenge
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to excess protein: {actual_protein} > {requirements['max_protein_per_serving']}")
                    continue
                else:
                    # If it meets the maximum protein requirement, we can consider it a valid contribution
                    individual_scan_contributions_pct.append(100)

            if 'max_calories_per_serving' in requirements and requirements['max_calories_per_serving'] > 0:
                if actual_calories > requirements['max_calories_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to excess calories: {actual_calories} > {requirements['max_calories_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'max_carbs_per_serving' in requirements and requirements['max_carbs_per_serving'] > 0:
                if actual_carbs > requirements['max_carbs_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to excess carbs: {actual_carbs} > {requirements['max_carbs_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)
            
            if 'max_fat_per_serving' in requirements and requirements['max_fat_per_serving'] > 0:
                if actual_fat > requirements['max_fat_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to excess fat: {actual_fat} > {requirements['max_fat_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'max_sugar_per_serving' in requirements and requirements['max_sugar_per_serving'] > 0:
                if actual_sugar > requirements['max_sugar_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to excess sugar: {actual_sugar} > {requirements['max_sugar_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            if 'max_sodium_per_serving' in requirements and requirements['max_sodium_per_serving'] > 0:
                if actual_sodium > requirements['max_sodium_per_serving']:
                    print(f"Skipping challenge {challenge.challengeId} for user {userid} due to excess sodium: {actual_sodium} > {requirements['max_sodium_per_serving']}")
                    continue
                else:
                    individual_scan_contributions_pct.append(100)

            


            if individual_scan_contributions_pct: # If this scan contributed to any relevant requirement
                # Calculate the average contribution of this scan across all its relevant requirements
                average_scan_contribution_pct = sum(individual_scan_contributions_pct) / len(individual_scan_contributions_pct)
                
                if average_scan_contribution_pct > 0: # Only update if there's a positive contribution
                    old_progress = accepted.progress
                    accepted.progress = min(100, int(accepted.progress + average_scan_contribution_pct))
                    if accepted.progress != old_progress:
                        progress_changed_in_this_update = True

            # Check for challenge completion
            # This condition is now based on the accumulated average of scan contributions.
            # It's an improvement but not a perfect guarantee that *all* individual total requirements are met.
            challenge_just_completed = False
            if accepted.progress >= 100 and not accepted.completed:
                accepted.completed = True
                accepted.progress = 100 # Cap progress at 100
                challenge_just_completed = True
                progress_changed_in_this_update = True # Mark change if it just got completed
                
                # Update leaderboard
                current_season = f"{current_time.year}-Q{(current_time.month-1)//3 + 1}"
                leaderboard = LeaderBoard.query.filter_by(
                    userId=userid,
                    season=current_season
                ).first()

                if not leaderboard:
                    leaderboard = LeaderBoard(
                        userId=userid,
                        season=current_season,
                        points=0
                    )
                    db.session.add(leaderboard)
                
                leaderboard.points += challenge.reward_points
                leaderboard.last_updated_date = current_time

            if progress_changed_in_this_update:
                db.session.commit()

    except Exception as e:
        db.session.rollback()
        print(f"Error updating challenge progress: {e}")
        raise e


def get_profile(userid):
    try:
        # Get the user
        user = db.session.get(User, userid)
        if not user:
            return None

        # Get today's date and date 7 days ago
        today = datetime.now().date()
        week_ago = today - timedelta(days=6)
        
        # Get all calories for the past 7 days in one query
        daily_calories = db.session.query(
            db.func.date(ScannedHistory.scanned_at).label('date'),
            db.func.sum(Food.calories_per_serving * (ScannedHistory.servings/Food.serving_size)).label('calories')
        ).join(
            Food, Food.foodId == ScannedHistory.foodId
        ).filter(
            ScannedHistory.userId == userid,
            ScannedHistory.scanned_at >= week_ago
        ).group_by(
            db.func.date(ScannedHistory.scanned_at)
        ).all()

        # Create a dictionary to store calories by date
        calories_by_date = {str(date): float(calories or 0) for date, calories in daily_calories}

        # Create ordered weekly calories with dates
        weekly_calories = []
        current_date = week_ago
        while current_date <= today:
            date_str = current_date.strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
            calories = calories_by_date.get(str(current_date), 0.0)
            weekly_calories.append((date_str, calories))
            current_date += timedelta(days=1)

        # Convert to ordered dict
        from collections import OrderedDict
        ordered_weekly_calories = OrderedDict(weekly_calories)

        # Get today's macros
        today_macros = db.session.query(
            db.func.sum(Food.protein_per_serving * ScannedHistory.servings),
            db.func.sum(Food.carbs_per_serving * ScannedHistory.servings),
            db.func.sum(Food.fat_per_serving * ScannedHistory.servings)
        ).join(
            ScannedHistory, Food.foodId == ScannedHistory.foodId
        ).filter(
            ScannedHistory.userId == userid,
            db.func.date(ScannedHistory.scanned_at) == today
        ).first()

        # Get current season's points and completed challenges
        current_season = f"{today.year}-Q{(today.month-1)//3 + 1}"
        leaderboard = LeaderBoard.query.filter_by(
            userId=userid,
            season=current_season
        ).first()

        completed_challenges = AcceptedChallenge.query.filter_by(
            userId=userid,
            completed=True
        ).count()

        # Compile profile data
        profile_data = {
            'personal_info': {
                'age': current_age(userid),
                'gender': user.gender,
                'height_cm': user.height_cm,
                'weight_kg': user.weight_kg,
                'activity_level': user.activity_level,
                'goal': user.goal,
                'dietary_pref': user.dietary_pref,
                'allergies': user.allergies,
                'medical_conditions': user.medical_conditions,
                'meal_times': user.meal_times,
                'address': user.address
            },
            'weekly_calories': ordered_weekly_calories,
            'today_macros': {
                'protein': float(today_macros[0] or 0),
                'carbs': float(today_macros[1] or 0),
                'fat': float(today_macros[2] or 0)
            },
            'achievements': {
                'season_points': leaderboard.points if leaderboard else 0,
                'completed_challenges': completed_challenges
            }
        }

        return profile_data

    except Exception as e:
        print(f"Error in get_profile: {str(e)}")
        return None

def current_age(userid):
    try:
        # Get the user
        user = db.session.get(User, userid)
        if not user:
            return None

        # Get today's date
        today = datetime.today().date()

        # Get stored age and join date
        stored_age = user.age  # Age when user joined
        join_date = user.joindate

        if join_date:
            # Calculate years passed since joining
            years_passed = today.year - join_date.year - ((today.month, today.day) < (join_date.month, join_date.day))
            # Current age is stored age plus years passed
            current_age = stored_age + years_passed
            return current_age
        else:
            return stored_age  # Return stored age if no join date

    except Exception as e:
        print(f"Error in current_age: {str(e)}")
        return None

def calculate_streak(userid):
    try:
        # Calculate current streak
        now = datetime.today()
        today = now.date()
        yesterday = today - timedelta(days=1)
        current_streak = 0

        # Check if user has scanned anything today
        today_scan = ScannedHistory.query.filter(
            ScannedHistory.userId == userid,
            db.func.date(ScannedHistory.scanned_at) == today
        ).first()

        # Get all user's scans ordered by date descending
        user_scans = ScannedHistory.query.filter(
            ScannedHistory.userId == userid,
            ScannedHistory.scanned_at < datetime.today()
        ).order_by(ScannedHistory.scanned_at.desc()).all()

        if user_scans:
            # Group scans by date
            scan_dates = set()
            for scan in user_scans:
                scan_dates.add(scan.scanned_at.date())
            scan_dates = sorted(list(scan_dates), reverse=True)

            if today_scan:
            # If there's a scan today, calculate consecutive days normally
                current_streak = 1
                for i in range(len(scan_dates) - 1):  # -1 because we already counted today
                    if scan_dates[i+1] == today - timedelta(days=i+1):
                        current_streak += 1
                    else:
                        break
            else:
                # If no scan today, check yesterday's scan
                yesterday_scan = next((scan for scan in user_scans if scan.scanned_at.date() == yesterday), None)
                
                if yesterday_scan:
                    # Calculate hours passed since last scan
                    hours_passed = (now - yesterday_scan.scanned_at).total_seconds() / 3600
                    
                    if hours_passed <= 24:
                        # Less than 24 hours passed, start counting from yesterday
                        current_streak = 1
                        for i in range(len(scan_dates) - 1):
                            if scan_dates[i+1] == yesterday - timedelta(days=i):
                                current_streak += 1
                            else:
                                break
                    else:
                        # More than 24 hours passed, reset streak
                        current_streak = 0
                else:
                    # No scan yesterday, streak is 0
                    current_streak = 0

        # Get user and update streak if current is longer
        user = db.session.get(User, userid)
        if user and current_streak > (user.lstreak or 0):
            user.lstreak = current_streak
            db.session.commit()

       # Create response data dictionary
        response_data = {
            'current_streak': current_streak,
            'longest_streak': user.lstreak if user else 0
        }

        return response_data

    except Exception as e:
        db.session.rollback()
        print(f"Error in calculate_streak: {str(e)}")
        raise e
    

def get_this_season_leaderboards(userid):
    try:
        # Get current season
        today = datetime.today()
        current_season = f"{today.year}-Q{(today.month-1)//3 + 1}"

        # Get all users' points for current season
        leaderboard_data = db.session.query(
            LeaderBoard.userId,
            LeaderBoard.points,
            User.username
        ).join(
            User, LeaderBoard.userId == User.userid
        ).filter(
            LeaderBoard.season == current_season
        ).order_by(
            LeaderBoard.points.desc()
        ).all()

        # Format response
        rankings = []
        for rank, (user_id, points, username) in enumerate(leaderboard_data, 1):
            rankings.append({
                'rank': rank,
                'username': username,
                'points': points,
                'isCurrentUser': user_id == userid
            })

        return rankings

    except Exception as e:
        print(f"Error in get_this_season_leaderboards: {str(e)}")
        return []


def get_this_week_leaderboards(userid):
    try:
        # Get start and end of current week
        today = datetime.today()
        start_of_week = today - timedelta(days=today.weekday())
        print(today.weekday())
        print(start_of_week)
        end_of_week = start_of_week + timedelta(days=6)
        print(end_of_week)

        # Get points accumulated this week
        leaderboard_data = db.session.query(
            AcceptedChallenge.userId,
            db.func.sum(Challenge.reward_points).label('points'),
            User.username
        ).join(
            Challenge, AcceptedChallenge.challengeId == Challenge.challengeId
        ).join(
            User, AcceptedChallenge.userId == User.userid
        ).filter(
            AcceptedChallenge.completed == True,
            db.func.date(AcceptedChallenge.accepted_date) >= start_of_week.date(),
            db.func.date(AcceptedChallenge.accepted_date) <= end_of_week.date()
        ).group_by(
            AcceptedChallenge.userId,
            User.username
        ).order_by(
            db.func.sum(Challenge.reward_points).desc()
        ).all()

        # Format response
        rankings = []
        for rank, (user_id, points, username) in enumerate(leaderboard_data, 1):
            rankings.append({
                'rank': rank,
                'username': username,
                'points': int(points or 0),
                'isCurrentUser': user_id == userid
            })

        return rankings

    except Exception as e:
        print(f"Error in get_this_week_leaderboards: {str(e)}")
        return []

def get_this_month_leaderboards(userid):
    try:
        # Get start and end of current month
        today = datetime.today()
        start_of_month = today.replace(day=1)
        next_month = today.replace(day=28) + timedelta(days=4)
        end_of_month = next_month - timedelta(days=next_month.day)

        # Get points accumulated this month
        leaderboard_data = db.session.query(
            AcceptedChallenge.userId,
            db.func.sum(Challenge.reward_points).label('points'),
            User.username
        ).join(
            Challenge, AcceptedChallenge.challengeId == Challenge.challengeId
        ).join(
            User, AcceptedChallenge.userId == User.userid
        ).filter(
            AcceptedChallenge.completed == True,
            db.func.date(AcceptedChallenge.accepted_date) >= start_of_month.date(),
            db.func.date(AcceptedChallenge.accepted_date) <= end_of_month.date()
        ).group_by(
            AcceptedChallenge.userId,
            User.username
        ).order_by(
            db.func.sum(Challenge.reward_points).desc()
        ).all()

        # Format response
        rankings = []
        for rank, (user_id, points, username) in enumerate(leaderboard_data, 1):
            rankings.append({
                'rank': rank,
                'username': username,
                'points': int(points or 0),
                'isCurrentUser': user_id == userid
            })

        return rankings

    except Exception as e:
        print(f"Error in get_this_month_leaderboards: {str(e)}")
        return []

@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory('FoodImage', filename)

##########################################################################################
def get_current_user_age(user_obj):
    """Calculates current age of the user."""
    if not user_obj or not user_obj.joindate or user_obj.age is None:
        return None # Or a default age like 25 if appropriate
    today = date.today()
    # Age at joining + years passed since joining
    years_passed = today.year - user_obj.joindate.year - \
                   ((today.month, today.day) < (user_obj.joindate.month, user_obj.joindate.day))
    return user_obj.age + years_passed

def calculate_nutritional_targets(user, age):
    """
    Calculates estimated BMR, TDEE, and target calories/macros.
    Uses Mifflin-St Jeor for BMR.
    """
    if not all([user.weight_kg, user.height_cm, age, user.gender is not None, user.activity_level is not None, user.goal is not None]):
        return None # Not enough info

    # BMR (Mifflin-St Jeor)
    bmr = (10 * user.weight_kg) + (6.25 * user.height_cm) - (5 * age)
    if user.gender == 1: # Male
        bmr += 5
    elif user.gender == -1: # Female
        bmr -= 161
    # else: Other gender, BMR calculation might need adjustment or use average. For now, no adjustment.

    # TDEE
    activity_factors = {-1: 1.2, 0: 1.55, 1: 1.725} # Sedentary, Moderate, Very Active
    activity_factor = activity_factors.get(user.activity_level, 1.375) # Default to lightly active if somehow invalid
    tdee = bmr * activity_factor

    # Calorie Goal Adjustment
    goal_calorie_adj = {-1: -400, 0: 0, 1: 400} # Slim, Maintain, Gain
    target_calories = tdee + goal_calorie_adj.get(user.goal, 0)

    # Macronutrient Targets (example: 40% Carbs, 30% Protein, 30% Fat)
    # Protein can also be g/kg, e.g., 1.6g/kg for active individuals
    # target_protein_grams = user.weight_kg * 1.6
    # target_protein_calories = target_protein_grams * 4
    # For simplicity with percentages:
    target_protein_calories = target_calories * 0.30
    target_carb_calories = target_calories * 0.40
    target_fat_calories = target_calories * 0.30

    return {
        "calories": round(target_calories),
        "protein": round(target_protein_calories / 4), # grams
        "carbs": round(target_carb_calories / 4),     # grams
        "fat": round(target_fat_calories / 9)         # grams
    }

def get_todays_intake(user_id):
    """Calculates total nutrients consumed by the user today."""
    today_start = datetime.combine(date.today(), datetime.min.time())
    today_end = datetime.combine(date.today(), datetime.max.time())

    scans_today = db.session.query(
        ScannedHistory.servings,
        Food.calories_per_serving, Food.protein_per_serving,
        Food.carbs_per_serving, Food.fat_per_serving, Food.sugar_per_serving,
        Food.sodium_per_serving, Food.serving_size, Food.foodId
    ).join(Food, ScannedHistory.foodId == Food.foodId)\
     .filter(
        ScannedHistory.userId == user_id,
        ScannedHistory.scanned_at >= today_start,
        ScannedHistory.scanned_at <= today_end
    ).all()

    intake = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "sugar": 0, "sodium": 0, "food_ids": set()}
    for scan in scans_today:
        servings_multiplier = scan.servings / (scan.serving_size if scan.serving_size and scan.serving_size > 0 else 100) # Avoid division by zero
        intake["calories"] += (scan.calories_per_serving or 0) * servings_multiplier
        intake["protein"] += (scan.protein_per_serving or 0) * servings_multiplier
        intake["carbs"] += (scan.carbs_per_serving or 0) * servings_multiplier
        intake["fat"] += (scan.fat_per_serving or 0) * servings_multiplier
        intake["sugar"] += (scan.sugar_per_serving or 0) * servings_multiplier
        intake["sodium"] += (scan.sodium_per_serving or 0) * servings_multiplier
        intake["food_ids"].add(scan.foodId)

    for key in ["calories", "protein", "carbs", "fat", "sugar", "sodium"]:
        intake[key] = round(intake[key])
    return intake

def get_excluded_elements_for_user(user):
    """
    Determines ingredients, allergens, and food tags to exclude based on user's
    allergies and medical conditions.
    Returns: dict with 'ingredient_ids', 'allergen_names_to_avoid', 'food_tags_to_avoid'
    """
    excluded = {
        "ingredient_ids": set(),
        "allergen_names_to_avoid": set(), # Store names for direct allergen check
        "food_tags_to_avoid": set()
    }

    # 1. User's direct allergies (assuming user.allergies is a list of allergen names)
    if user.allergies and isinstance(user.allergies, list):
        for allergen_name_from_user in user.allergies:
            allergen_name_lower = allergen_name_from_user.lower().strip()
            print(f"Processing allergen: {allergen_name_lower}")
            excluded["allergen_names_to_avoid"].add(allergen_name_lower)
            # Find ingredients associated with this allergen
            allergen_obj = Allergen.query.filter(func.lower(Allergen.name) == allergen_name_lower).first()
            if allergen_obj:
                for ingredient in allergen_obj.ingredients:
                    excluded["ingredient_ids"].add(ingredient.ingredient_id)
                    print(f"Adding ingredient ID {ingredient.name} for allergen {allergen_name_lower}")

    # 2. Medical Condition Guidelines
    if user.medical_conditions and isinstance(user.medical_conditions, list):
        for condition_name_from_user in user.medical_conditions:
            condition_name_lower = condition_name_from_user.lower().strip()
            guidelines = MedicalConditionDietaryGuideline.query.filter(
                func.lower(MedicalConditionDietaryGuideline.condition_name) == condition_name_lower
            ).all()
            print(f"Processing medical condition: {condition_name_lower} with {len(guidelines)} guidelines")

            for g in guidelines:
                target_lower = g.parameter_target.lower()
                if g.guideline_type == "AVOID_INGREDIENT_NAME" and g.parameter_target_type == "INGREDIENT":
                    ingredient = Ingredient.query.filter(func.lower(Ingredient.name) == target_lower).first()
                    if ingredient:
                        excluded["ingredient_ids"].add(ingredient.ingredient_id)
                        print(f"Adding ingredient ID {ingredient.name} for condition {condition_name_lower}")
                if g.guideline_type == "AVOID_ALLERGEN_NAME" and g.parameter_target_type == "ALLERGEN":
                    excluded["allergen_names_to_avoid"].add(target_lower)
                    allergen_obj = Allergen.query.filter(func.lower(Allergen.name) == target_lower).first()
                    if allergen_obj:
                        for ingredient in allergen_obj.ingredients:
                            excluded["ingredient_ids"].add(ingredient.ingredient_id)
                            print(f"Adding ingredient ID {ingredient.name} for allergen {target_lower}")
                if g.guideline_type == "AVOID_FOOD_TAG" and g.parameter_target_type == "TAG":
                    excluded["food_tags_to_avoid"].add(target_lower)
    return excluded

def get_user_food_interactions_maps(user_id):
    """Fetches user's ratings, cooked, and bookmarked food IDs."""
    interactions = UserFoodInteraction.query.filter_by(userId=user_id).all()
    ratings = {i.foodId: i.rating for i in interactions if i.rating is not None}
    cooked = {i.foodId for i in interactions if i.has_been_cooked}
    bookmarked = {i.foodId for i in interactions if i.is_bookmarked}
    return {"ratings": ratings, "cooked": cooked, "bookmarked": bookmarked}

# --- Main Recommendation Function ---

def recommendation(user_id, num_recommendations=10):
    user = db.session.get(User, user_id)
    if not user:
        app.logger.error(f"Recommendation: User with ID {user_id} not found.")
        return []

    age = get_current_user_age(user)
    if age is None: # Handle case where age cannot be determined
        app.logger.warning(f"Recommendation: Could not determine age for user {user_id}. Using default or skipping age-dependent calcs.")
        # Decide on a fallback or return early if age is critical

    targets = calculate_nutritional_targets(user, age if age is not None else 25) # Use default age if None
    if not targets:
        app.logger.error(f"Recommendation: Could not calculate nutritional targets for user {user_id}.")
        return []

    todays_intake = get_todays_intake(user_id)
    excluded_elements = get_excluded_elements_for_user(user)
    user_interactions = get_user_food_interactions_maps(user_id)

    remaining_needs = {
        "calories": targets["calories"] - todays_intake["calories"],
        "protein": targets["protein"] - todays_intake["protein"],
        "carbs": targets["carbs"] - todays_intake["carbs"],
        "fat": targets["fat"] - todays_intake["fat"],
    }

    all_foods = Food.query.all()
    candidate_foods = []

    for food in all_foods:
        score = 0
        is_excluded = False

        # Priority 1: Exclusions
        # Check food tags to avoid
        if food.tags and isinstance(food.tags, list):
            if any(tag.lower() in excluded_elements["food_tags_to_avoid"] for tag in food.tags):
                is_excluded = True
                # app.logger.debug(f"Excluding food {food.name} due to excluded tag.")
                continue

        # Check ingredients to avoid (via FoodItemIngredient)
        food_ingredient_ids = {fii.ingredientId for fii in FoodItemIngredient.query.filter_by(foodId=food.foodId).all()}
        app.logger.debug(f"Food {food.name} has ingredients: {food_ingredient_ids}")
        if not excluded_elements["ingredient_ids"].isdisjoint(food_ingredient_ids):
            is_excluded = True
            app.logger.debug(f"Excluding food {food.name} due to excluded ingredient ID.")
            continue
        
        # Check allergens to avoid (more complex if food doesn't directly list its allergens)
        # This part relies on ingredients being correctly linked to allergens.
        # If a food contains an ingredient that belongs to an "allergen_names_to_avoid", it's covered by ingredient_ids.
        # A direct check could be: if food has an "allergen" field and it matches. (Not in current Food model)

        if is_excluded:
            continue

        # Priority 3 (Dietary Preference - can act as a strong filter)
        if user.dietary_pref:
            pref_lower = user.dietary_pref.lower()
            food_tags_lower = [t.lower() for t in food.tags] if food.tags else []
            
            if pref_lower == "vegetarian":
                if "vegan" not in food_tags_lower and "vegetarian" not in food_tags_lower: # If not explicitly tagged
                    # Basic check for meat in name/category (can be improved)
                    if "meat" in (food.category.lower() if food.category else "") or \
                       any(m_word in food.name.lower() for m_word in ["chicken", "beef", "pork", "lamb", "fish", "seafood"]):
                        score -= 1000 # Effectively exclude
            elif pref_lower == "vegan":
                if "vegan" not in food_tags_lower:
                     score -= 1000 # Effectively exclude

        if score <= -1000: # Hard exclusion by preference
            continue

        # Priority 2: Nutritional Fit & Recent Intake
        # Calories: Aim for foods that help meet remaining needs without overshooting too much
        # This is a simple heuristic, can be much more sophisticated
        calories_per_serving = food.calories_per_serving or 0
        if remaining_needs["calories"] > 0:
            if 0.1 * remaining_needs["calories"] < calories_per_serving < 0.7 * remaining_needs["calories"]:
                score += 30 # Good fit for remaining calories
            elif calories_per_serving > remaining_needs["calories"] * 1.2: # Significantly over
                score -= 20
        elif calories_per_serving > 300 : # If calorie needs met, penalize high calorie items
             score -= 15


        # Protein: Prioritize if protein is needed
        protein_per_serving = food.protein_per_serving or 0
        if remaining_needs["protein"] > 10 and protein_per_serving > 15: # Needs protein, food is good source
            score += 25
        elif remaining_needs["protein"] < -10 and protein_per_serving > 20: # Already over protein, penalize very high protein
            score -= 10

        # Similar logic for carbs and fat can be added.

        # Penalty for already eaten today
        if food.foodId in todays_intake["food_ids"]:
            score -= 15

        # Priority 3: User Interactions & Preferences (continued)
        if user.dietary_pref and food.tags:
            pref_lower = user.dietary_pref.lower()
            food_tags_lower = [t.lower() for t in food.tags]
            if pref_lower in food_tags_lower:
                score += 20 # Matches general dietary preference tag

        if food.foodId in user_interactions["ratings"]:
            score += user_interactions["ratings"][food.foodId] * 8 # Rating 1-5 -> 8-40 points
        if food.foodId in user_interactions["cooked"]:
            score += 10
        if food.foodId in user_interactions["bookmarked"]:
            score += 5
        
        # Add food and its score to candidates
        candidate_foods.append({"food": food, "score": score})

    # Sort candidates by score in descending order
    sorted_foods = sorted(candidate_foods, key=lambda x: x["score"], reverse=True)

    # Format output
    recommendations_output = []
    for item in sorted_foods[:num_recommendations]:
        f = item["food"]
        recommendations_output.append({
            "foodId": f.foodId,
            "name": f.name,
            "description": f.description
            # "debug_score": item["score"] # Optional: for debugging
        })
    
    app.logger.info(f"Generated {len(recommendations_output)} recommendations for user {user_id}.")
    return recommendations_output

# --- Recommendation Endpoint ---
@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    try:
        num_recs_str = request.args.get('count', '10')
        try:
            num_recs = int(num_recs_str)
            if not (1 <= num_recs <= 50): # Limit count
                num_recs = 10
        except ValueError:
            num_recs = 10

        recs = recommendation(user_id, num_recommendations=num_recs)
        if not recs:
            return create_response("success", 200, "No specific recommendations could be generated at this time, or user data is insufficient.", [])
        return create_response("success", 200, f"Successfully retrieved {len(recs)} food recommendations.", recs)
    except Exception as e:
        app.logger.error(f"Error in /recommendations/{user_id}: {str(e)}", exc_info=True)
        return create_response("error", 500, "An error occurred while generating recommendations.", str(e))

#######################################################################################################

def recommendationByCopilot(userId, num_recommendations=10):
    try:
        # Get user and validate
        user = db.session.get(User, userId)
        if not user:
            return []

        # Priority 1: Medical Conditions and Allergies
        excluded_foods = set()
        
        # Check allergies
        if user.allergies:
            allergen_ingredients = set()
            for allergen_name in user.allergies:
                allergen = Allergen.query.filter(db.func.lower(Allergen.name) == allergen_name.lower()).first()
                if allergen:
                    for ingredient in allergen.ingredients:
                        allergen_ingredients.add(ingredient.ingredient_id)
            
            # Get foods containing allergenic ingredients
            allergenic_foods = db.session.query(Food.foodId).join(
                FoodItemIngredient,
                Food.foodId == FoodItemIngredient.foodId
            ).filter(
                FoodItemIngredient.ingredientId.in_(allergen_ingredients)
            ).all()
            
            excluded_foods.update([f[0] for f in allergenic_foods])

        # Priority 2: Health Metrics and Today's Intake
        # Calculate BMI
        height_m = user.height_cm / 100
        bmi = user.weight_kg / (height_m * height_m) if height_m > 0 else 0

        # Get today's intake
        today = datetime.today()
        today_scans = db.session.query(
            db.func.sum(Food.calories_per_serving * ScannedHistory.servings).label('calories'),
            db.func.sum(Food.protein_per_serving * ScannedHistory.servings).label('protein')
        ).join(
            Food, ScannedHistory.foodId == Food.foodId
        ).filter(
            ScannedHistory.userId == userId,
            db.func.date(ScannedHistory.scanned_at) == today.date()
        ).first()

        total_calories_today = float(today_scans.calories or 0)
        total_protein_today = float(today_scans.protein or 0)

        # Calculate target calories based on metrics
        base_calories = 2000  # Base value
        if user.activity_level == 1:  # Very Active
            base_calories *= 1.5
        elif user.activity_level == -1:  # Not Active
            base_calories *= 0.8

        if user.goal == 1:  # Gain Weight
            target_calories = base_calories * 1.2
        elif user.goal == -1:  # Lose Weight
            target_calories = base_calories * 0.8
        else:
            target_calories = base_calories

        remaining_calories = target_calories - total_calories_today

        # Query all foods
        foods = Food.query.filter(~Food.foodId.in_(excluded_foods)).all()
        scored_foods = []

        # Get user's food interactions
        interactions = UserFoodInteraction.query.filter_by(userId=userId).all()
        ratings = {i.foodId: i.rating for i in interactions if i.rating}

        for food in foods:
            score = 0
            
            # Priority 2: Health Metrics Scoring
            calories_fit = abs(food.calories_per_serving - (remaining_calories / 3))  # Assume 3 meals
            score += (1000 - calories_fit) / 100  # Better fit = higher score

            if user.goal == 1 and food.protein_per_serving > 20:  # Weight gain, prioritize protein
                score += 30
            elif user.goal == -1 and food.calories_per_serving < 400:  # Weight loss, prioritize low-cal
                score += 30

            # Priority 3: Dietary Preferences and Ratings
            if user.dietary_pref and food.tags:
                if user.dietary_pref.lower() in [tag.lower() for tag in food.tags]:
                    score += 20

            if food.foodId in ratings:
                score += ratings[food.foodId] * 5

            scored_foods.append((food, score))

        # Sort by score and get top recommendations
        scored_foods.sort(key=lambda x: x[1], reverse=True)
        recommendations = []
        for food, _ in scored_foods[:num_recommendations]:
            recommendations.append({
                'foodId': food.foodId,
                'name': food.name,
                'description': food.description
            })

        return recommendations

    except Exception as e:
        print(f"Error in recommendationByCopilot: {str(e)}")
        return []

# Endpoint
@app.route('/recommendations/copilot/<int:userId>', methods=['GET'])
def get_copilot_recommendations(userId):
    try:
        recommendations = recommendationByCopilot(userId)
        if not recommendations:
            return create_response(
                "success", 
                200, 
                "No recommendations available or user not found.", 
                []
            )
        
        return create_response(
            "success",
            200,
            f"Successfully retrieved {len(recommendations)} recommendations.",
            recommendations
        )
    except Exception as e:
        return create_response(
            "error",
            500,
            "An error occurred while generating recommendations.",
            str(e)
        )

#######################################################################################################
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # only creates tables if they don’t exist
    app.run(debug=True,port=8000)

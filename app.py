from flask import Flask, jsonify, request
from sqlalchemy.exc import IntegrityError
from config import Config
from models import db, User,Challenge
from datetime import date,datetime

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

@app.route('/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@app.route('/users', methods=['POST'])
def add_user():
    # Extract individual parameters from form data
    password = request.form.get('password')
    email = request.form.get('email')
    username = request.form.get('username')
    joinDate = date.today()
    age = request.form.get('age', type=int)
    gender = request.form.get('gender')
    height_cm = request.form.get('height_cm', type=float)
    weight_kg = request.form.get('weight_kg', type=float)
    activity_level = request.form.get('activity_level', type=int)
    goal = request.form.get('goal', type=int)
    dietary_pref = request.form.get('dietary_pref')
    allergies = request.form.get('allergies')  # JSON string
    medical_conditions = request.form.get('medical_conditions')  # JSON string
    meal_times = request.form.get('meal_times')  # JSON string
    address = request.form.get('address')

    if not username or not email or not password:  # Example validation
        return jsonify({'error': 'Username, email and password are required'}), 400

    try:

        # Parse JSON fields if provided
        allergies = eval(allergies) if allergies else None
        medical_conditions = eval(medical_conditions) if medical_conditions else None
        meal_times = eval(meal_times) if meal_times else None


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
        return jsonify({'message': 'User added successfully'}), 201
    
    except IntegrityError:
        db.session.rollback()  # Rollback the transaction in case of an error
        return jsonify({'error': 'Username or email already exists'}), 400
    

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return jsonify({'error': str(e)}), 500
    
@app.route('/challenges', methods=['GET'])
def get_challenge():
    challenges = Challenge.query.all()
    return jsonify([challenge.to_dict() for challenge in challenges])

@app.route('/challenges', methods=['POST'])
def add_challenge():
    # Extract parameters from the request
    title = request.form.get('title')
    description = request.form.get('description')
    reward_points = request.form.get('reward_points', type=int)
    deadline = request.form.get('deadline')  # Expected in ISO format (e.g., "2025-04-20T23:59:59")

    # Validate required fields
    if not title or not reward_points or not deadline:
        return jsonify({'error': 'Title, reward points, and deadline are required'}), 400

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
        return jsonify({'message': 'Challenge added successfully', 'challenge': new_challenge.to_dict()}), 201

    except Exception as e:
        db.session.rollback()  # Rollback the transaction in case of an error
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # only creates tables if they don’t exist
    app.run(debug=True,port=8000)

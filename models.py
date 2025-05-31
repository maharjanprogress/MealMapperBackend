from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import CheckConstraint, UniqueConstraint, ForeignKey
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    userid = db.Column(db.Integer, primary_key=True)
    password = db.Column(db.String)        # You can specify length like db.String(128) if known
    email = db.Column(db.String, unique=True, nullable=False)
    username = db.Column(db.String, unique=True, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.Integer)           # -1 for female, 0 for others, 1 for male
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    activity_level = db.Column(db.Integer)  # -1 for sedentary, 0 for moderate, 1 for active
    goal = db.Column(db.Integer)         # -1 for weight loss, 0 for maintenance, 1 for weight gain
    dietary_pref = db.Column(db.String)
    allergies = db.Column(JSON)
    medical_conditions = db.Column(JSON)
    meal_times = db.Column(JSON)
    joindate = db.Column(db.Date)      # Or db.Date if it's just date
    address = db.Column(db.String)
    lstreak = db.Column(db.Integer, default=0)  # Streak of days user has logged in


    __table_args__ = (
        CheckConstraint("goal IN (-1, 0, 1)", name="valid_goal_constraint"),
        CheckConstraint("gender IN (-1, 0, 1)", name="valid_gender_constraint"),
        CheckConstraint("activity_level IN (-1, 0, 1)", name="valid_activity_level_constraint"),
    )

    def to_dict(self):
        return {
            'userId': self.userid,
            'username': self.username,
            'email': self.email,
            "age": self.age,
            "gender": self.gender,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "activity_level": self.activity_level,
            "goal": self.goal,
            "dietary_pref": self.dietary_pref,
            "allergies": self.allergies,
            "medical_conditions": self.medical_conditions,
            "meal_times": self.meal_times,
            'joindate': self.joindate.isoformat() if self.joindate else None,
            'address': self.address,
            'streak': self.lstreak

            # ⚠️ Not including password for security reasons
        }
    
class Challenge(db.Model):
    __tablename__ = 'challenge'

    challengeId = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    reward_points = db.Column(db.Integer, nullable=False)
    deadline = db.Column(db.DateTime, nullable=False)
    difficulty = db.Column(db.String, nullable=True)
    requirements = db.Column(JSON, nullable=False)  # New column added

    def to_dict(self):
        return {
            'challengeId': self.challengeId,
            'title': self.title,
            'description': self.description,
            'reward_points': self.reward_points,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'difficulty': self.difficulty,
            'requirements': self.requirements

        }

class AcceptedChallenge(db.Model):
    __tablename__ = 'accepted_challenge'

    id = db.Column(db.Integer, primary_key=True)
    challengeId = db.Column(db.Integer, db.ForeignKey('challenge.challengeId', ondelete='CASCADE'), nullable=False)
    userId = db.Column(db.Integer, db.ForeignKey('users.userid', ondelete='CASCADE'), nullable=False)
    progress = db.Column(db.Integer, default=0, nullable=False)
    completed = db.Column(db.Boolean, default=False)
    accepted_date = db.Column(db.DateTime, default=lambda: datetime.now(datetime.timezone.utc))

    # Define relationship with Challenge
    challenge = db.relationship('Challenge', backref='accepted_challenges', lazy=True)
    
    __table_args__ = (
        CheckConstraint("progress >= 0 AND progress <= 100", name="valid_progress_constraint"),
        UniqueConstraint('challengeId', 'userId', name='unique_challenge_user')
    )

    def to_dict(self):
        return {
            'id': self.id,
            'challengeId': self.challengeId,
            'userId': self.userId,
            'progress': self.progress,
            'completed': self.completed,
            'accepted_date': self.accepted_date.isoformat() if self.accepted_date else None
        }

class LeaderBoard(db.Model):
    __tablename__ = 'leader_board'

    leaderId = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('users.userid', ondelete='CASCADE'), nullable=False)
    season = db.Column(db.String(10), nullable=False)  # e.g., '2025-Q2'
    points = db.Column(db.Integer, default=0)
    last_updated_date = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('userId', 'season', name='unique_user_season'),
    )

    def to_dict(self):
        return {
            'leaderId': self.leaderId,
            'userId': self.userId,
            'season': self.season,
            'points': self.points,
            'last_updated_date': self.last_updated_date.isoformat() if self.last_updated_date else None
        }
    
class Food(db.Model):
    __tablename__ = 'foods'

    foodId = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique = True)
    description = db.Column(db.Text)
    calories_per_serving = db.Column(db.Float, nullable=False)
    protein_per_serving = db.Column(db.Float, default=0)
    carbs_per_serving = db.Column(db.Float, default=0)
    fat_per_serving = db.Column(db.Float, default=0)
    sugar_per_serving = db.Column(db.Float, default=0)
    sodium_per_serving = db.Column(db.Float, nullable=True, default=None)
    serving_size = db.Column(db.Float(50), default=100) # Default serving size is 100g
    category = db.Column(db.String(50))
    meal_type = db.Column(db.String(20), nullable=False)  # ✅ Now required
    tags = db.Column(JSON)
    contents = db.Column(JSON)
    recipe = db.Column(JSON)
    image_url = db.Column(db.Text)
    popularity_score = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        CheckConstraint(
            "meal_type IN ('breakfast', 'lunch', 'dinner')",
            name="valid_meal_type_constraint"
        ),
    )

    def to_dict(self):
        return {
            'foodId': self.foodId,
            'name': self.name,
            'description': self.description,
            'calories_per_serving': self.calories_per_serving,
            'protein_per_serving': self.protein_per_serving,
            'carbs_per_serving': self.carbs_per_serving,
            'fat_per_serving': self.fat_per_serving,
            'sugar_per_serving': self.sugar_per_serving,
            'sodium_per_serving': self.sodium_per_serving,
            'serving_size': self.serving_size,
            'category': self.category,
            'meal_type': self.meal_type,
            'tags': self.tags,
            'contents': self.contents,
            'recipe': self.recipe,
            'image_url': self.image_url,
            'popularity_score': self.popularity_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def to_dictFind(self):
        return {
            'name': self.name,
            'meal_type': self.meal_type,
            'serving_size': self.serving_size
        }

class ScannedHistory(db.Model):
    __tablename__ = 'scanned_history'

    scanId = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('users.userid', ondelete='CASCADE'), nullable=False)
    foodId = db.Column(db.Integer, db.ForeignKey('foods.foodId', ondelete='CASCADE'), nullable=False)
    scanned_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    meal_time = db.Column(db.String(20), nullable=False)  # breakfast, lunch, dinner
    servings = db.Column(db.Float, default=1.0)  # How many servings were consumed

    # Define relationships
    user = db.relationship('User', backref='scanned_histories', lazy=True)
    food = db.relationship('Food', backref='scan_records', lazy=True)

    __table_args__ = (
        CheckConstraint(
            "meal_time IN ('breakfast', 'lunch', 'dinner')",
            name="valid_meal_time_constraint"
        ),
        CheckConstraint("servings > 0", name="valid_servings_constraint"),
    )

    def to_dict(self):
        return {
            'scanId': self.scanId,
            'userId': self.userId,
            'foodId': self.foodId,
            'scanned_at': self.scanned_at.isoformat() if self.scanned_at else None,
            'meal_time': self.meal_time,
            'servings': self.servings
        }
    
# Association table for the many-to-many relationship between Ingredient and Allergen
ingredient_allergen_association = db.Table('ingredient_allergen_association',
    db.Column('ingredient_id', db.Integer, db.ForeignKey('ingredients.ingredient_id', ondelete='CASCADE'), primary_key=True),
    db.Column('allergen_id', db.Integer, db.ForeignKey('allergens.allergen_id', ondelete='CASCADE'), primary_key=True)
)

class Allergen(db.Model):
    __tablename__ = 'allergens'

    allergen_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False) # e.g., "Gluten", "Dairy", "Peanuts"
    description = db.Column(db.Text, nullable=True)

    # Relationship: An allergen can be linked to many ingredients
    # Changed to a many-to-many relationship
    ingredients = db.relationship(
        'Ingredient',
        secondary=ingredient_allergen_association,
        back_populates='allergens',
        lazy='dynamic'  # Or 'select' / 'joined' based on access patterns
    )


    def to_dict(self):
        return {
            'allergen_id': self.allergen_id,
            'name': self.name,
            'description': self.description,
            'associated_ingredients': [{'ingredient_id': ingredient.ingredient_id, 'name': ingredient.name} for ingredient in self.ingredients]
        }

class Ingredient(db.Model):
    __tablename__ = 'ingredients'

    ingredient_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False) # e.g., "Whole Wheat Flour", "Cow's Milk"
    # allergen_id = db.Column(db.Integer, db.ForeignKey('allergens.allergen_id'), nullable=True) # REMOVED: Replaced by many-to-many

    # New many-to-many relationship to Allergen
    allergens = db.relationship(
        'Allergen',
        secondary=ingredient_allergen_association,
        back_populates='ingredients',
        lazy='dynamic' # Or 'select' / 'joined'
    )

    def to_dict(self):
        return {
            'ingredient_id': self.ingredient_id,
            'name': self.name,
            'associated_allergens': [{'allergen_id': allergen.allergen_id, 'name': allergen.name} for allergen in self.allergens]
        }

class FoodItemIngredient(db.Model):
    __tablename__ = 'food_item_ingredients' # Changed from FoodIngredients for clarity

    food_item_ingredient_id = db.Column(db.Integer, primary_key=True)
    foodId = db.Column(db.Integer, db.ForeignKey('foods.foodId', ondelete='CASCADE'), nullable=False)
    ingredientId = db.Column(db.Integer, db.ForeignKey('ingredients.ingredient_id', ondelete='CASCADE'), nullable=False)
    quantity_description = db.Column(db.String(100), nullable=True) # e.g., "1 cup", "100g", "to taste"
    notes = db.Column(db.Text, nullable=True) # e.g., "finely chopped"

    # Relationships
    food = db.relationship('Food', backref=db.backref('food_ingredients_association', lazy=True, cascade="all, delete-orphan"))
    ingredient = db.relationship('Ingredient', backref=db.backref('ingredient_in_foods_association', lazy=True))

    __table_args__ = (
        UniqueConstraint('foodId', 'ingredientId', name='unique_food_ingredient'),
    )

    def to_dict(self):
        return {
            'food_item_ingredient_id': self.food_item_ingredient_id,
            'foodId': self.foodId,
            'ingredientId': self.ingredientId,
            'ingredient_name': self.ingredient.name if self.ingredient else None, # For convenience
            'quantity_description': self.quantity_description,
            'notes': self.notes
        }

class MedicalConditionDietaryGuideline(db.Model):
    __tablename__ = 'medical_condition_dietary_guidelines'

    guideline_id = db.Column(db.Integer, primary_key=True)
    condition_name = db.Column(db.String(255), nullable=False, index=True) # e.g., "Hypertension", "Type 2 Diabetes"
    guideline_type = db.Column(db.String(100), nullable=False)
    # Examples: "AVOID_INGREDIENT_NAME", "AVOID_ALLERGEN_NAME", "LIMIT_NUTRIENT", "PREFER_FOOD_TAG"
    parameter_target = db.Column(db.String(255), nullable=False)
    # Examples: "Salt" (ingredient), "Gluten" (allergen), "Sodium" (nutrient), "low-sodium" (tag)
    parameter_target_type = db.Column(db.String(50), nullable=False)
    # Examples: "INGREDIENT", "ALLERGEN", "NUTRIENT", "TAG"
    threshold_value = db.Column(db.Float, nullable=True) # e.g., 2000 (for sodium mg/day)
    threshold_unit = db.Column(db.String(50), nullable=True) # e.g., "mg/day", "g/serving"
    severity = db.Column(db.String(50), nullable=True) # e.g., "Strict Avoid", "Limit", "Recommended"
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'guideline_id': self.guideline_id,
            'condition_name': self.condition_name,
            'guideline_type': self.guideline_type,
            'parameter_target': self.parameter_target,
            'parameter_target_type': self.parameter_target_type,
            'threshold_value': self.threshold_value,
            'threshold_unit': self.threshold_unit,
            'severity': self.severity,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class UserFoodInteraction(db.Model):
    __tablename__ = 'user_food_interactions'

    interaction_id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('users.userid', ondelete='CASCADE'), nullable=False)
    foodId = db.Column(db.Integer, db.ForeignKey('foods.foodId', ondelete='CASCADE'), nullable=False)

    is_bookmarked = db.Column(db.Boolean, default=True, nullable=False)
    # 'state' from your request: False if only bookmarked, True if cooked
    has_been_cooked = db.Column(db.Boolean, default=False, nullable=False)

    rating = db.Column(db.Integer, nullable=True) # e.g., 1-5
    notes = db.Column(db.Text, nullable=True) # User's personal notes about the food/recipe

    bookmarked_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    cooked_at = db.Column(db.DateTime, nullable=True) # Timestamp when marked as cooked
    last_updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = db.relationship('User', backref=db.backref('food_interactions', lazy='dynamic'))
    food = db.relationship('Food', backref=db.backref('user_interactions', lazy='dynamic'))

    __table_args__ = (
        UniqueConstraint('userId', 'foodId', name='unique_user_food_interaction'),
        CheckConstraint("rating IS NULL OR (rating >= 1 AND rating <= 5)", name="valid_rating_constraint"),
    )

    def to_dict(self):
        return {
            'interaction_id': self.interaction_id,
            'userId': self.userId,
            'foodId': self.foodId,
            'is_bookmarked': self.is_bookmarked,
            'has_been_cooked': self.has_been_cooked,
            'rating': self.rating,
            'notes': self.notes,
            'bookmarked_at': self.bookmarked_at.isoformat() if self.bookmarked_at else None,
            'cooked_at': self.cooked_at.isoformat() if self.cooked_at else None,
            'last_updated_at': self.last_updated_at.isoformat() if self.last_updated_at else None
        }

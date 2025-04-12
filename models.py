from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import CheckConstraint

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
            'address': self.address

            # ⚠️ Not including password for security reasons
        }

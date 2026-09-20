"""
CareerCompass AI Dataset Generator
Generates a comprehensive, realistic dataset of student profiles, academic backgrounds,
skills, and mapped optimal career paths with multi-dimensional skill evaluations.
"""

import os
import random
import pandas as pd

CAREER_ARCHETYPES = [
    # Technology & AI
    {
        "career": "Software Developer",
        "field": "Technology",
        "subjects": ["Computer Science", "Engineering", "Mathematics"],
        "min_skill": "Beginner",
        "skills_pool": ["Python", "Java", "C++", "Data Structures", "Algorithms", "Git", "Problem Solving", "Object-Oriented Programming", "REST APIs", "Debugging"],
        "demand": "High Demand",
        "scores": (85, 65, 88, 70, 60),
    },
    {
        "career": "Full-Stack Developer",
        "field": "Technology",
        "subjects": ["Computer Science", "Engineering"],
        "min_skill": "Intermediate",
        "skills_pool": ["JavaScript", "HTML/CSS", "React", "Node.js", "Python", "SQL", "MongoDB", "Express", "REST APIs", "Git"],
        "demand": "High Demand",
        "scores": (88, 68, 85, 75, 62),
    },
    {
        "career": "AI Engineer",
        "field": "Technology",
        "subjects": ["Artificial Intelligence", "Computer Science", "Mathematics", "Data Science"],
        "min_skill": "Intermediate",
        "skills_pool": ["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow", "Linear Algebra", "Calculus", "NLP", "Neural Networks", "Algorithms"],
        "demand": "High Demand",
        "scores": (92, 70, 90, 80, 65),
    },
    {
        "career": "Data Scientist",
        "field": "Data Science & AI",
        "subjects": ["Data Science", "Mathematics", "Computer Science", "Economics"],
        "min_skill": "Intermediate",
        "skills_pool": ["Python", "R", "SQL", "Pandas", "Scikit-Learn", "Statistics", "Data Visualization", "Machine Learning", "Tableau", "Analytical Thinking"],
        "demand": "High Demand",
        "scores": (90, 75, 92, 72, 65),
    },
    {
        "career": "Cybersecurity Analyst",
        "field": "Technology",
        "subjects": ["Cybersecurity", "Computer Science", "Engineering"],
        "min_skill": "Intermediate",
        "skills_pool": ["Network Security", "Penetration Testing", "Cryptography", "Linux", "Ethical Hacking", "Wireshark", "Firewalls", "Python", "Incident Response"],
        "demand": "High Demand",
        "scores": (88, 65, 87, 65, 60),
    },
    {
        "career": "Cloud Engineer",
        "field": "Technology",
        "subjects": ["Computer Science", "Engineering"],
        "min_skill": "Intermediate",
        "skills_pool": ["AWS", "Azure", "Docker", "Kubernetes", "Linux", "Terraform", "CI/CD", "Networking", "Python", "System Architecture"],
        "demand": "High Demand",
        "scores": (87, 66, 84, 62, 63),
    },
    {
        "career": "Mobile App Developer",
        "field": "Technology",
        "subjects": ["Computer Science", "Engineering", "Design"],
        "min_skill": "Intermediate",
        "skills_pool": ["Flutter", "React Native", "Swift", "Kotlin", "Android Studio", "UI/UX", "Mobile Architecture", "REST APIs", "Git"],
        "demand": "Trending",
        "scores": (86, 68, 82, 80, 62),
    },
    {
        "career": "DevOps Engineer",
        "field": "Technology",
        "subjects": ["Computer Science", "Engineering"],
        "min_skill": "Intermediate",
        "skills_pool": ["Docker", "Kubernetes", "Jenkins", "GitLab CI", "Linux Bash", "Python", "Cloud Security", "Infrastructure as Code", "Monitoring"],
        "demand": "High Demand",
        "scores": (89, 67, 86, 60, 65),
    },
    # Healthcare & Science
    {
        "career": "Doctor",
        "field": "Healthcare",
        "subjects": ["Biology", "Medical Science", "Chemistry"],
        "min_skill": "None",
        "skills_pool": ["Patient Care", "Clinical Diagnosis", "Anatomy", "Physiology", "Pharmacology", "Emergency Response", "Empathy", "Critical Thinking", "Medical Ethics"],
        "demand": "High Demand",
        "scores": (88, 88, 92, 60, 82),
    },
    {
        "career": "Dentist",
        "field": "Healthcare",
        "subjects": ["Medical Science", "Biology", "Chemistry"],
        "min_skill": "None",
        "skills_pool": ["Oral Surgery", "Dental Anatomy", "Patient Communication", "Diagnostics", "Manual Dexterity", "Hygiene Standards", "Empathy"],
        "demand": "Stable",
        "scores": (85, 82, 84, 70, 70),
    },
    {
        "career": "Pharmacist",
        "field": "Healthcare",
        "subjects": ["Chemistry", "Medical Science", "Biology"],
        "min_skill": "None",
        "skills_pool": ["Pharmacology", "Drug Interactions", "Organic Chemistry", "Dosage Calculation", "Patient Counseling", "Attention to Detail", "Quality Control"],
        "demand": "Stable",
        "scores": (84, 76, 82, 58, 65),
    },
    {
        "career": "Psychologist",
        "field": "Healthcare",
        "subjects": ["Psychology", "Biology"],
        "min_skill": "None",
        "skills_pool": ["Active Listening", "Cognitive Behavioral Therapy", "Empathy", "Research Methods", "Mental Health Assessment", "Counseling", "Ethics"],
        "demand": "High Demand",
        "scores": (72, 94, 86, 75, 78),
    },
    {
        "career": "Biomedical Engineer",
        "field": "Healthcare",
        "subjects": ["Biology", "Engineering", "Physics"],
        "min_skill": "Beginner",
        "skills_pool": ["Biomechanics", "Medical Devices", "Signal Processing", "Prosthetics", "CAD Design", "Python", "Biology", "Materials Science"],
        "demand": "Trending",
        "scores": (88, 70, 88, 78, 66),
    },
    {
        "career": "Research Scientist",
        "field": "Science",
        "subjects": ["Physics", "Chemistry", "Biology", "Mathematics"],
        "min_skill": "Beginner",
        "skills_pool": ["Scientific Research", "Hypothesis Testing", "Data Analysis", "Academic Writing", "Laboratory Techniques", "Critical Analysis", "Python", "Statistics"],
        "demand": "Stable",
        "scores": (90, 72, 94, 76, 68),
    },
    {
        "career": "Astrophysicist",
        "field": "Science",
        "subjects": ["Physics", "Mathematics"],
        "min_skill": "Beginner",
        "skills_pool": ["Astrophysics", "Quantum Mechanics", "Calculus", "Numerical Simulation", "Python", "Cosmology", "Data Modeling", "Astronomical Observation"],
        "demand": "Emerging",
        "scores": (93, 68, 95, 74, 62),
    },
    # Engineering
    {
        "career": "Mechanical Engineer",
        "field": "Engineering",
        "subjects": ["Engineering", "Physics", "Mathematics"],
        "min_skill": "None",
        "skills_pool": ["SolidWorks", "AutoCAD", "Thermodynamics", "Fluid Mechanics", "Robotics", "Manufacturing", "Material Science", "Problem Solving"],
        "demand": "Stable",
        "scores": (86, 68, 88, 72, 68),
    },
    {
        "career": "Civil Engineer",
        "field": "Engineering",
        "subjects": ["Engineering", "Physics", "Mathematics"],
        "min_skill": "None",
        "skills_pool": ["Structural Analysis", "AutoCAD", "Project Planning", "Surveying", "Geotechnical Engineering", "Construction Management", "Site Inspection"],
        "demand": "Stable",
        "scores": (85, 72, 86, 65, 75),
    },
    {
        "career": "Electrical Engineer",
        "field": "Engineering",
        "subjects": ["Engineering", "Physics", "Mathematics"],
        "min_skill": "Beginner",
        "skills_pool": ["Circuit Design", "Power Systems", "Microcontrollers", "MATLAB", "Embedded Systems", "PCB Design", "Signal Processing"],
        "demand": "High Demand",
        "scores": (88, 66, 89, 70, 66),
    },
    {
        "career": "Aerospace Engineer",
        "field": "Engineering",
        "subjects": ["Engineering", "Physics", "Mathematics"],
        "min_skill": "Beginner",
        "skills_pool": ["Aerodynamics", "Propulsion Systems", "Avionics", "CAD Modeling", "Flight Simulation", "Structural Analysis", "MATLAB"],
        "demand": "Emerging",
        "scores": (91, 68, 92, 76, 70),
    },
    {
        "career": "Robotics Engineer",
        "field": "Engineering",
        "subjects": ["Engineering", "Computer Science", "Artificial Intelligence", "Physics"],
        "min_skill": "Intermediate",
        "skills_pool": ["ROS", "C++", "Python", "Kinematics", "Sensors & Actuators", "Computer Vision", "Control Systems", "Mechatronics"],
        "demand": "High Demand",
        "scores": (92, 68, 91, 82, 68),
    },
    # Design & Creative Arts
    {
        "career": "UI/UX Designer",
        "field": "Design",
        "subjects": ["Design", "Computer Science", "Psychology"],
        "min_skill": "None",
        "skills_pool": ["Figma", "User Research", "Wireframing", "Prototyping", "Design Thinking", "Interaction Design", "Usability Testing", "Visual Hierarchy", "Design Systems"],
        "demand": "High Demand",
        "scores": (78, 82, 80, 94, 68),
    },
    {
        "career": "Graphic Designer",
        "field": "Design",
        "subjects": ["Design", "Animation"],
        "min_skill": "None",
        "skills_pool": ["Photoshop", "Illustrator", "Typography", "Branding", "Visual Identity", "Color Theory", "Creativity", "Layout Design"],
        "demand": "Trending",
        "scores": (75, 76, 74, 95, 62),
    },
    {
        "career": "Animator",
        "field": "Design",
        "subjects": ["Animation", "Design"],
        "min_skill": "None",
        "skills_pool": ["Blender", "Maya", "3D Modeling", "Character Animation", "Storyboarding", "Motion Graphics", "After Effects", "Texturing"],
        "demand": "Emerging",
        "scores": (80, 68, 76, 96, 60),
    },
    {
        "career": "Product Designer",
        "field": "Design",
        "subjects": ["Design", "Engineering", "Business Studies"],
        "min_skill": "None",
        "skills_pool": ["Product Strategy", "Figma", "Industrial Design", "User Empathy", "Prototyping", "Customer Research", "Iterative Design"],
        "demand": "High Demand",
        "scores": (80, 84, 85, 92, 75),
    },
    # Business, Finance & Marketing
    {
        "career": "Chartered Accountant",
        "field": "Finance",
        "subjects": ["Commerce", "Finance", "Economics"],
        "min_skill": "None",
        "skills_pool": ["Financial Accounting", "Taxation", "Auditing", "Financial Reporting", "Excel Mastery", "Corporate Law", "Financial Analysis", "Attention to Detail"],
        "demand": "Stable",
        "scores": (88, 74, 90, 55, 72),
    },
    {
        "career": "Investment Banker",
        "field": "Finance",
        "subjects": ["Finance", "Economics", "Commerce", "Mathematics"],
        "min_skill": "None",
        "skills_pool": ["Financial Modeling", "Valuation", "M&A Analysis", "Excel", "Pitch Decks", "Capital Markets", "Market Research", "Negotiation"],
        "demand": "High Demand",
        "scores": (88, 86, 91, 62, 85),
    },
    {
        "career": "Financial Analyst",
        "field": "Finance",
        "subjects": ["Finance", "Economics", "Mathematics", "Commerce"],
        "min_skill": "None",
        "skills_pool": ["Financial Statement Analysis", "Forecasting", "Excel", "PowerBI", "Market Research", "Ratio Analysis", "Python for Finance"],
        "demand": "Trending",
        "scores": (86, 78, 89, 60, 68),
    },
    {
        "career": "Business Analyst",
        "field": "Business",
        "subjects": ["Business Studies", "Economics", "Computer Science"],
        "min_skill": "Beginner",
        "skills_pool": ["Requirements Gathering", "SQL", "Tableau", "Process Mapping", "Agile/Scrum", "Business Strategy", "Stakeholder Management", "Data Analytics"],
        "demand": "High Demand",
        "scores": (82, 86, 88, 70, 78),
    },
    {
        "career": "Product Manager",
        "field": "Business",
        "subjects": ["Business Studies", "Computer Science", "Economics"],
        "min_skill": "Beginner",
        "skills_pool": ["Product Vision", "Roadmap Planning", "User Research", "Metrics & KPIs", "Leadership", "Cross-Functional Coordination", "Agile", "Market Strategy"],
        "demand": "Trending",
        "scores": (80, 92, 89, 82, 92),
    },
    {
        "career": "Entrepreneur",
        "field": "Business",
        "subjects": ["Business Studies", "Commerce", "Economics", "Computer Science"],
        "min_skill": "None",
        "skills_pool": ["Business Model Generation", "Pitching & Fundraising", "Risk Management", "Growth Strategy", "Sales & Marketing", "Leadership", "Resilience"],
        "demand": "Emerging",
        "scores": (75, 94, 90, 88, 96),
    },
    {
        "career": "Digital Marketer",
        "field": "Marketing",
        "subjects": ["Business Studies", "Commerce", "Economics"],
        "min_skill": "None",
        "skills_pool": ["SEO", "SEM", "Google Analytics", "Social Media Marketing", "Content Strategy", "Email Campaigns", "Copywriting", "PPC Advertising"],
        "demand": "High Demand",
        "scores": (78, 88, 80, 86, 74),
    },
    # Law & Public Service
    {
        "career": "Lawyer",
        "field": "Law",
        "subjects": ["Law", "Psychology"],
        "min_skill": "None",
        "skills_pool": ["Legal Research", "Argumentation", "Constitutional Law", "Courtroom Advocacy", "Contract Drafting", "Public Speaking", "Negotiation", "Critical Thinking"],
        "demand": "Stable",
        "scores": (76, 96, 91, 72, 85),
    },
    {
        "career": "IAS Officer",
        "field": "Government",
        "subjects": ["Law", "Economics", "Education"],
        "min_skill": "None",
        "skills_pool": ["Public Administration", "Policy Formulation", "Crisis Management", "General Studies", "Governance", "Leadership", "Public Communication", "Decision Making"],
        "demand": "Stable",
        "scores": (80, 92, 92, 70, 95),
    },
    {
        "career": "Journalist",
        "field": "Marketing",
        "subjects": ["Education", "Law", "Psychology"],
        "min_skill": "None",
        "skills_pool": ["Investigative Reporting", "Interviewing", "News Writing", "Media Ethics", "Fact Checking", "Storytelling", "Video Journalism"],
        "demand": "Stable",
        "scores": (70, 95, 84, 85, 72),
    },
    # Education
    {
        "career": "Professor",
        "field": "Education",
        "subjects": ["Education", "Physics", "Chemistry", "Mathematics", "Computer Science"],
        "min_skill": "None",
        "skills_pool": ["Academic Lecturing", "Curriculum Development", "Scientific Publication", "Mentorship", "Public Speaking", "Research Methodology", "Pedagogy"],
        "demand": "Stable",
        "scores": (85, 92, 88, 76, 82),
    },
    {
        "career": "Teacher",
        "field": "Education",
        "subjects": ["Education", "Mathematics", "Physics", "Chemistry", "Biology"],
        "min_skill": "None",
        "skills_pool": ["Classroom Management", "Lesson Planning", "Child Psychology", "Student Evaluation", "Patience", "Empathy", "Communication"],
        "demand": "Stable",
        "scores": (75, 94, 80, 80, 80),
    },
    # Aviation & Hospitality
    {
        "career": "Pilot",
        "field": "Aviation",
        "subjects": ["Physics", "Mathematics", "Engineering"],
        "min_skill": "None",
        "skills_pool": ["Flight Navigation", "Aircraft Systems", "Meteorology", "Spatial Awareness", "Emergency Procedures", "Radio Communication", "High Focus"],
        "demand": "Trending",
        "scores": (88, 82, 92, 55, 84),
    },
    {
        "career": "Chef",
        "field": "Hospitality",
        "subjects": ["Hospitality", "Chemistry"],
        "min_skill": "None",
        "skills_pool": ["Culinary Techniques", "Menu Engineering", "Food Safety", "Creativity", "Time Management", "Kitchen Leadership", "Sensory Evaluation"],
        "demand": "Trending",
        "scores": (78, 74, 80, 92, 80),
    }
]

SEMESTERS = [
    "10th", "12th", "Diploma", "Graduate", "Postgraduate",
    "1st Semester", "2nd Semester", "3rd Semester", "4th Semester",
    "5th Semester", "6th Semester", "7th Semester", "8th Semester"
]

PROGRAMMING_SKILLS = ["None", "Beginner", "Intermediate", "Advanced", "Expert"]

def generate_student_samples(num_samples=1200, seed=42):
    random.seed(seed)
    data = []

    for _ in range(num_samples):
        archetype = random.choice(CAREER_ARCHETYPES)
        career = archetype["career"]
        demand = archetype["demand"]
        subject = random.choice(archetype["subjects"])
        
        # Skill level
        min_skill = archetype["min_skill"]
        if min_skill == "Intermediate":
            skill_choice = random.choices(["Intermediate", "Advanced", "Expert", "Beginner"], weights=[0.45, 0.35, 0.15, 0.05])[0]
        elif min_skill == "Beginner":
            skill_choice = random.choices(["Beginner", "Intermediate", "Advanced", "None"], weights=[0.45, 0.35, 0.15, 0.05])[0]
        else:
            skill_choice = random.choices(["None", "Beginner", "Intermediate", "Advanced"], weights=[0.55, 0.25, 0.15, 0.05])[0]

        # Age and Semester
        age = random.randint(17, 28)
        if age <= 18:
            semester = random.choice(["10th", "12th", "1st Semester"])
        elif age <= 22:
            semester = random.choice(["2nd Semester", "3rd Semester", "4th Semester", "5th Semester", "6th Semester", "7th Semester", "8th Semester", "Diploma"])
        else:
            semester = random.choice(["Graduate", "Postgraduate", "7th Semester", "8th Semester"])

        # Sample 3-6 skills from the archetype pool plus 0-2 general soft skills
        num_core_skills = random.randint(3, min(6, len(archetype["skills_pool"])))
        sampled_skills = random.sample(archetype["skills_pool"], num_core_skills)
        
        general_skills = ["Teamwork", "Adaptability", "Communication", "Time Management", "Critical Thinking", "Problem Solving", "Creativity"]
        extra = random.sample(general_skills, random.randint(0, 2))
        combined_skills = list(set(sampled_skills + extra))
        skills_text = ", ".join(combined_skills)

        # Career interest: mostly matches or related
        if random.random() < 0.85:
            interest = career
        else:
            # Pick an interest in the same subject area
            rel_careers = [a["career"] for a in CAREER_ARCHETYPES if any(s in a["subjects"] for s in archetype["subjects"])]
            interest = random.choice(rel_careers)

        # Skill scores with realistic Gaussian jitter
        base_t, base_c, base_p, base_cr, base_l = archetype["scores"]
        
        # Adjust technical score based on programming skill
        tech_boost = {"None": -5, "Beginner": 0, "Intermediate": 5, "Advanced": 10, "Expert": 14}[skill_choice]
        
        tech_score = max(45, min(99, int(round(random.gauss(base_t + tech_boost, 4)))))
        comm_score = max(45, min(99, int(round(random.gauss(base_c, 5)))))
        prob_score = max(45, min(99, int(round(random.gauss(base_p, 4)))))
        creat_score = max(45, min(99, int(round(random.gauss(base_cr, 5)))))
        lead_score = max(40, min(99, int(round(random.gauss(base_l, 6)))))

        data.append({
            "age": age,
            "semester": semester,
            "favorite_subject": subject,
            "programming_skill": skill_choice,
            "career_interest": interest,
            "skills_text": skills_text,
            "target_career": career,
            "career_demand": demand,
            "technical_score": tech_score,
            "communication_score": comm_score,
            "problem_solving_score": prob_score,
            "creativity_score": creat_score,
            "leadership_score": lead_score,
        })

    df = pd.DataFrame(data)
    return df

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "career_dataset.csv")

    print(f"Generating synthetic AI training dataset...")
    df = generate_student_samples(num_samples=1800, seed=2026)
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {len(df)} samples saved to: {csv_path}")
    print("Class distribution preview:")
    print(df["target_career"].value_counts().head(10))

if __name__ == "__main__":
    main()


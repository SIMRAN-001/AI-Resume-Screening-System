"""
Skill dictionary: canonical skill name -> list of synonyms / spellings.

Add your own skills here. Matching is case-insensitive and whole-token, so
"java" will not match inside "javascript".
"""

SKILLS = {
    # Programming languages
    "python": ["python", "python3"],
    "java": ["java"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "go": ["golang"],
    "r": ["r programming", "r language", "rstudio"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql", "pl/sql"],
    "scala": ["scala"],
    "bash": ["bash", "shell scripting"],
    # Data / ML
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "opencv"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling", "hypothesis testing"],
    "data analysis": ["data analysis", "data analytics", "exploratory data analysis", "eda"],
    "data visualization": ["data visualization", "data visualisation"],
    "feature engineering": ["feature engineering"],
    "time series": ["time series", "time-series", "forecasting"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow"],
    "keras": ["keras"],
    "pytorch": ["pytorch", "torch"],
    "xgboost": ["xgboost", "lightgbm", "gradient boosting"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib", "seaborn"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "microsoft excel", "advanced excel"],
    "spark": ["spark", "pyspark", "apache spark"],
    "hadoop": ["hadoop", "hive"],
    "airflow": ["airflow", "apache airflow"],
    "etl": ["etl", "data pipelines", "data pipeline"],
    "mlops": ["mlops", "mlflow", "model deployment", "kubeflow"],
    "llm": ["llm", "llms", "large language models", "generative ai", "langchain"],
    # Web / backend
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular"],
    "node.js": ["node.js", "nodejs", "node js"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "spring boot": ["spring boot", "spring"],
    "html/css": ["html", "css", "html5", "css3"],
    "rest api": ["rest api", "rest apis", "restful", "rest"],
    "microservices": ["microservices"],
    # DevOps / cloud
    "aws": ["aws", "amazon web services", "ec2", "s3", "sagemaker"],
    "azure": ["azure"],
    "gcp": ["gcp", "google cloud"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab"],
    "ci/cd": ["ci/cd", "jenkins", "github actions", "gitlab ci"],
    "linux": ["linux", "unix"],
    # Databases
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "nosql": ["nosql", "cassandra", "dynamodb"],
    # Other
    "agile": ["agile", "scrum", "kanban"],
    "communication": ["communication skills", "stakeholder management"],
    "leadership": ["team lead", "leadership", "mentoring"],
}

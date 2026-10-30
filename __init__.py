"""Source package for the Resume-Job Gap Analyzer.

Each module has one job:

    resume_parser    -> get raw text out of an uploaded file
    text_cleaner     -> turn messy raw text into tidy text
    skill_extractor  -> find known skills inside tidy text
    matcher          -> compare resume skills with job skills, score the match
    recommendations  -> turn the gaps into ranked, actionable advice
"""

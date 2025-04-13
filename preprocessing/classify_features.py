def classify_features():
    """
    Returns a dictionary with feature classification.
    """
    return {
        'stimulants': ['ORB', 'AST', 'STL', 'BLK', 'PTS', 'FG%', '3P%', 'FT%'],
        'destimulants': ['TOV'],
        'nominants': []
    }
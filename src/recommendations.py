# generates study tips based on scores
def generate_recommendations(data):
    recs = []
    
    # math tips
    if data['math score'] < 60:
        recs.append("Math: Focus on fundamental concepts and practice daily.")
    elif data['math score'] < 80:
        recs.append("Math: Good progress! Try solving more advanced logic puzzles.")
    
    # reading tips
    if data['reading score'] < 60:
        recs.append("Reading: Increase reading time. Focus on non-fiction articles.")
    elif data['reading score'] < 80:
        recs.append("Reading: Solid performance. Engage in more complex literature.")
        
    # writing tips
    if data['writing score'] < 60:
        recs.append("Writing: Practice essay writing and check your grammar.")
    elif data['writing score'] < 80:
        recs.append("Writing: Well done. Work on creative writing.")
        
    # overall feedback
    avg = (data['math score'] + data['reading score'] + data['writing score']) / 3
    
    if avg < 50:
        recs.append("Overall: You should talk to an advisor for a catch-up plan.")
    elif avg < 70:
        recs.append("Overall: Just stay consistent with your study schedule.")
    else:
        recs.append("Overall: Excellent! Consider honors-level material.")
        
    return recs

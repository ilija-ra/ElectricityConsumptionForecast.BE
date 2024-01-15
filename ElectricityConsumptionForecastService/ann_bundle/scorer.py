class Scorer:
    def get_score(self, y_test, y_predicted):
        return abs((y_test - y_predicted)/y_test)*100
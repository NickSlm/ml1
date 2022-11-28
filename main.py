from utils import fetch_housing_data,load_housing_data
import matplotlib.pyplot as plt

def main():
    # fetch_housing_data()

    data_frame = load_housing_data()
    data_frame_id = data_frame.reset_index()
    

if __name__ == "__main__":
    main()

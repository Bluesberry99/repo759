import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("processed_data.csv")  # Replace with your local file path

# Combine Month, Day, Hour, and Minute columns into a datetime column
data["Datetime"] = pd.to_datetime({
    "year": 2024,  # Assuming year is fixed
    "month": data["Month"],
    "day": data["Day"],
    "hour": data["Hour"],
    "minute": data["Minute"]
})

# Identify the first occurrence of each month
month_starts = data.groupby("Month").first()  # Get the first row for each month
month_start_indices = month_starts.index  # Indices of first rows for each month
month_start_labels = month_starts["Datetime"].dt.strftime("%b %d")  # Format as 'Month Day'

# Plot the Returns over time
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Returns"], label="Returns", color="blue")

# Customize the plot
plt.title("Returns Over Time with Monthly Markers", fontsize=14)
plt.xlabel("Time Steps (Minutes)", fontsize=12)
plt.ylabel("Returns", fontsize=12)

# Set custom X-axis ticks for the first day of each month
plt.xticks(month_starts.index, month_start_labels, rotation=45)  # Rotate labels for readability
plt.legend()
plt.grid(True)

# # Save the plot to a file
plt.savefig("simulated_returns_ with_Markers.png")

# Show the plot
#plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data from the CSV file
# data = pd.read_csv("processed_data.csv")

# # Extract the "Returns" column
# time = data.index  # Use the row index as time steps
# returns = data["Returns"]  # Extract the Returns column for plotting

# # Plot the Returns over time
# plt.figure(figsize=(12, 6))
# plt.plot(time, returns, label="Simulated Returns", color="blue")

# # Customize the plot
# plt.title("Simulated Returns Over Time", fontsize=14)
# plt.xlabel("Time Steps (Index)", fontsize=12)
# plt.ylabel("Returns", fontsize=12)
# #plt.legend()
# plt.grid(True)

# # Save the plot to a file
# plt.savefig("simulated_returns.png")

# # Show the plot
# #plt.show()

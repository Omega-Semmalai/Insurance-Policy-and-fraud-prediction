import matplotlib.pyplot as plt
import networkx as nx
import os

# Create a directed graph
G = nx.DiGraph()

# Define user roles and their navigation paths
nodes = [
    "Login Page",
    "Administrator",
    "User Management",
    "Room Management (CRUD)",
    "Rent Management",
    "Reporting",
    "Staff",
    "Room Management (View)",
    "Allotee Management (View)",
    "Rent Management (View)",
    "Reporting (View)"
]

edges = [
    ("Login Page", "Administrator"),
    ("Login Page", "Staff"),
    ("Administrator", "User Management"),
    ("Administrator", "Room Management (CRUD)"),
    ("Administrator", "Rent Management"),
    ("Administrator", "Reporting"),
    ("Staff", "Room Management (View)"),
    ("Staff", "Allotee Management (View)"),
    ("Staff", "Rent Management (View)"),
    ("Staff", "Reporting (View)"),
]

# Add nodes and edges to the graph
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Create a layout for the nodes
pos = nx.spring_layout(G, seed=42)  # positions for all nodes

# Draw the nodes and edges
plt.figure(figsize=(10, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True, arrowstyle='-|>', arrowsize=20)
plt.title("User-Wise Navigation Diagram for Hostel Management System")
plt.grid("on")

# Save the diagram as an image in the specified directory
directory = "C:\\Omega\\Semester 5\\Machine Learning\\Project"
os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
image_path = os.path.join(directory, "User_Wise_Navigation_Diagram.png")
plt.savefig(image_path, format="png")
plt.close()

image_path

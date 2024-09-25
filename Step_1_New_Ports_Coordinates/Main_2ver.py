import pyvista as pv
import numpy as np
import Tools.Function_for_DATA_angel as FuD
import coordinate_output_NBI_and_ports as Cout

def create_surface(R_x, R_y, R_z):
    points, faces = [], []
    for i in range(len(R_x) - 1):
        current_contour = np.column_stack((R_x[i], R_y[i], R_z[i]))
        next_contour = np.column_stack((R_x[i+1], R_y[i+1], R_z[i+1]))
        for j in range(len(current_contour)):
            p1, p2 = current_contour[j], current_contour[(j+1) % len(current_contour)]
            n1, n2 = next_contour[j], next_contour[(j+1) % len(current_contour)]
            points.extend([p1, p2, n2, n1])
            faces.append([4, len(points)-4, len(points)-3, len(points)-2, len(points)-1])
    return pv.PolyData(np.array(points), faces=np.hstack(faces))

def find_intersection(surface, point, direction):
    ray_end = point + direction / np.linalg.norm(direction) * 1000
    return surface.ray_trace(point, ray_end)[0][0]

def main():
    # Data for surface and ports
    R_x, R_y, R_z = FuD.all_point(FuD.read_data()[0])
    P_1, P_2 = np.array(Cout.Ports()[:2])
    
    # Create 3D surface
    surface = create_surface(R_x, R_y, R_z)
    plotter = pv.Plotter()
    plotter.add_mesh(surface, color='cyan', show_edges=True)

    # Find intersections and create new P_1 array
    new_P_1 = []  
    lines = []

    for i in range(P_1.shape[1]):
        start_point = P_1[:, i]
        direction_vector = P_2[:, i] - start_point
        
        # Find intersection point
        intersection = find_intersection(surface, start_point, direction_vector)
        
        # Append intersection point coordinates to new_P_1
        new_P_1.append(intersection) 

        # Create a line from the start point to the intersection point
        line = pv.Line(start_point, intersection)
        lines.append(line)
    
    # Add all the lines to the plotter
    for line in lines:
        plotter.add_mesh(line, color='yellow', line_width=3)
    
    plotter.show()
    new_P_1 = np.array(new_P_1).T
    return new_P_1  # Return the new P_1 structure

if __name__ == "__main__":
    new_P_1 = main()
    print(new_P_1)

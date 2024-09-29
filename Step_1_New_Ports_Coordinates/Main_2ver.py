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

def find_first_two_intersections(surface, point, direction):
    ray_end = point + direction / np.linalg.norm(direction) * 1000
    intersections = surface.ray_trace(point, ray_end)[0]
    return intersections[0], intersections[1]  


def get_intersection_points_NBI(NBI_X, NBI_Y, NBI_Z,NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z, surface):
    new_NBI_start = []  
    new_NBI_end = []
    lines = []
    NBI_P1 = np.array([NBI_X, NBI_Y, NBI_Z])
    NBI_P2 = np.array([NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z])
  
    for i in range(NBI_P1.shape[1]):
        start_point = NBI_P1[:, i]
        direction_vector = NBI_P2[:, i] 

        # Find intersection point
        intersection1, intersection2 = find_first_two_intersections(surface, start_point, direction_vector)
        new_NBI_start.append(intersection1) 
        new_NBI_end.append(intersection2) 
        line = pv.Line(start_point, intersection2)
        lines.append(line)
    

    new_NBI_start = np.array(new_NBI_start).T
    new_NBI_end = np.array(new_NBI_end).T

    return new_NBI_start, new_NBI_end, lines

def get_intersection_points(P_1,P_2, surface):
    new_P_1 = []  
    lines = []
    for i in range(P_1.shape[1]):
        start_point = P_1[:, i]
        direction_vector = P_2[:, i] - start_point
        
        # Find intersection point
        intersection = find_intersection(surface, start_point, direction_vector)
        new_P_1.append(intersection) 
        
        line = pv.Line(start_point, intersection)
        lines.append(line)

    new_P_1 = np.array(new_P_1).T
    return new_P_1, lines




if __name__ == "__main__":
    # Data for surface and ports
    R_x, R_y, R_z = FuD.all_point(FuD.read_data()[0])
    P_1, P_2 = np.array(Cout.Ports()[:2])

    #Downnload Input data of coordinates ports and NBI
    NBI_X, NBI_Y, NBI_Z, NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z = Cout.NBI()
    

    # Create 3D surface
    surface = create_surface(R_x, R_y, R_z)
    plotter = pv.Plotter()
    plotter.add_mesh(surface, color='cyan', show_edges=True)
    

    #get new P_2 
    new_P_1, lines1 = get_intersection_points(P_1,P_2, surface)
    new_NBI_start, new_NBI_end, lines2 = get_intersection_points_NBI(NBI_X, NBI_Y, NBI_Z,NBI_uvec_X, NBI_uvec_Y, NBI_uvec_Z, surface)

    for line in lines1:
        plotter.add_mesh(line, color='yellow', line_width=3)
    

    for line in lines2:
        plotter.add_mesh(line, color='red', line_width=3)


    plotter.show()









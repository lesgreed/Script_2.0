import pyvista as pv
import numpy as np
import Tools.Function_for_DATA_angel as FuD

def create_and_save_torus_surface(R_x, R_y, R_z, output_filename='torus_surface.obj'):

    all_contours = []

  
    for section_index in range(len(R_x)):
        section_points = np.column_stack((R_x[section_index], R_y[section_index], R_z[section_index]))

        n_points = len(section_points)
        lines = np.array([n_points] + list(range(n_points)) + [0], dtype=np.int32)  

        contour_lines = pv.PolyData(section_points, lines=lines)
        
        all_contours.append(contour_lines)

    combined_contours = pv.PolyData()
    for contour in all_contours:
        combined_contours += contour


    surface = combined_contours.extract_surface() 


    plotter = pv.Plotter()
    plotter.add_mesh(surface, color='cyan')
    plotter.add_title('Toroidal Surface')
    plotter.show()
    surface.save(output_filename)
    print(f'Torus surface saved to {output_filename}')

if __name__ == "__main__":

    Phi, R_phi, Z_phi = FuD.read_data()
    R_x_all, R_y_all, Z_all = FuD.all_point(Phi)
    R_x, R_y, R_z = R_x_all, R_y_all, Z_all


    create_and_save_torus_surface(R_x, R_y, R_z)

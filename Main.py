import tkinter as tk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import numpy as np
import Surface_rays as geo
import Surface_data as FuD
import NBI_Ports_data_input as Cout
import J_0_test.mconf.mconf as mconf
import os 

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("W7-X cheking")
        self.geometry(f"{1520}x{900}")
        self.data_instance = Data()
        self.Bget = calculus()

        # Global variable for slider value
        self.angle = int(90)
        self.scale = 10  # New global variable for the second slider

        # Section 1: Sidebar
        self.create_sidebar()

        # Section 3: Additional Widgets
        self.create_additional_widgets()

        # Initialize default values
        self.nbi_optionmenu.set(self.nbi_options[0])
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.update_port_options(self.nbi_options[0])

        self.current_graph = None
        self.all_results = []

    def create_sidebar(self):
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(2, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="W7-X cheking", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))



        self.show_old_button_label= ctk.CTkLabel(self.sidebar_frame, text="Raw data:", anchor="w")    
        self.show_old_button_label.grid(row=4, column=0, padx=10, pady=(10,0))   
        self.show_old_button = ctk.CTkButton(self.sidebar_frame, text="Update data", command=lambda: self.pre_calculate())
        self.show_old_button.grid(row=4, column=0, padx=10, pady=(70,0))
    

    def pre_calculate(self):
        if len(self.all_results) ==0:
         Result_array = self.data_instance.data_already_input()  
         self.all_results = Result_array
        else:
            self.all_results = self.all_results[:2]
        print(len(self.all_results[1][2]))

        time = datetime.now().strftime("%H:%M:%S")
        self.textbox.insert("end", f"\n\n [{time}]: Old data ready \n\n ")
        
    def create_additional_widgets(self):
        # create textbox
        self.textbox = ctk.CTkTextbox(self, width=30)  
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Button to show graph
        self.show_graph_button = ctk.CTkButton(self, text="Add Port", command=lambda: self.dummy_function())
        self.show_graph_button.grid(row=1, column=1, padx=(20, 0), pady=(10, 0), sticky="w")


        # create tabview with two sections
        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Section 1: "PORTS and NBI"
        self.tabview.add("PORTS and NBI")
        self.tabview.tab("PORTS and NBI").grid_columnconfigure(0, weight=1)

        # First CTkOptionMenu for NBI selection
        self.nbi_options = self.generate_nbi_options()
        self.nbi_optionmenu_label = ctk.CTkLabel(self.tabview.tab("PORTS and NBI"), text="Select NBI or Gyrotron Launcher", anchor="w")
        self.nbi_optionmenu_label.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        self.nbi_optionmenu = ctk.CTkOptionMenu(self.tabview.tab("PORTS and NBI"), values=self.nbi_options, command=self.update_port_options)
        self.nbi_optionmenu.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Second CTkOptionMenu for Port selection (based on NBI selection)
        self.port_options = []
        self.port_optionmenu_label = ctk.CTkLabel(self.tabview.tab("PORTS and NBI"), text="Select Port", anchor="w")
        self.port_optionmenu_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.port_optionmenu = ctk.CTkOptionMenu(self.tabview.tab("PORTS and NBI"), values=self.port_options)
        self.port_optionmenu.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")

        # Section 2: "Setting"
        self.tabview.add("Setting")
        self.tabview.tab("Setting").grid_columnconfigure(0, weight=1)

        # Slider in "Setting" section
        self.slider_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Adjust Value:", anchor="w")
        self.slider_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        self.slider = ctk.CTkSlider(self.tabview.tab("Setting"), from_=30, to=90, command=self.slider_event)
        self.slider.set(self.angle)
        self.slider.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="w")

        # Label to display the current slider value
        self.value_label = ctk.CTkLabel(self.tabview.tab("Setting"), text=str(self.angle))
        self.value_label.grid(row=1, column=1, padx=(10, 20), pady=(10, 20), sticky="w")

        # Second slider for values 10 to 100 without intermediate values
        self.second_slider_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Adjust Second Value:", anchor="w")
        self.second_slider_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        self.second_slider = ctk.CTkSlider(self.tabview.tab("Setting"), from_=1, to=10, command=self.second_slider_event)
        self.second_slider.set(self.scale // 10)  # Set initial value based on second_slider_value
        self.second_slider.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="w")

        # Label to display the current value of the second slider
        self.second_value_label = ctk.CTkLabel(self.tabview.tab("Setting"), text=str(self.scale))
        self.second_value_label.grid(row=3, column=1, padx=(10, 20), pady=(10, 20), sticky="w")

        # Canvas for displaying the graph
        self.graph_canvas = ctk.CTkCanvas(self, width=800, height=600, bg="white")
        self.graph_canvas.grid(row=0, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
    
    def second_slider_event(self, value):
        self.scale = int(value) * 10  
        self.second_value_label.configure(text=str(self.scale))


    def slider_event(self, value):
        self.angle = int(value)
        self.value_label.configure(text=str(self.angle))


    def generate_port_options(self, selected_nbi: str):
        nbi_index = int(selected_nbi.split("_")[1])
        if selected_nbi.startswith("CTS"):
         nbi_index = nbi_index+8
        
        nbi_index = nbi_index-1
        valid_indices, extreme_points_1, extreme_points_2, valid_port_names = self.data_instance.port_for_nbi(nbi_index, int(self.angle), self.scale)
        Ports_for_NBI_Index = valid_port_names
        return Ports_for_NBI_Index 

    def update_port_options(self, selected_nbi: str):
     self.port_options = self.generate_port_options(selected_nbi)
     self.port_optionmenu.configure(values=self.port_options)  # Оновлення варіантів
    # Set the initial value for Select Port (if the list is not empty)
     if self.port_options:
            self.port_optionmenu.set(self.port_options[0])
    

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

    def generate_nbi_options(self):
      return [f"NBI_{i}" if i <= 8 else f"CTS_{i-8}" for i in range(1, 12)]



    def generate_and_show_graph(self):
        # Example function to generate and display the graph
        # Clear previous graph
        if self.current_graph:
            self.current_graph.get_tk_widget().destroy()

        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        # Embed the figure in the canvas
        canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="nsew")

        # Update current graph
        self.current_graph = canvas

    def dummy_function(self):
        # Example function for button interaction
        selected_nbi = self.nbi_optionmenu.get()
        selected_port = self.port_optionmenu.get()
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"[{timestamp}]: Selected Port: {selected_port}\nSelected NBI: {selected_nbi}\n\n"
        self.textbox.insert("end", message)

        # Call the function to generate the graph
        self.generate_and_show_graph()
class Data:
    def __init__(self):
        self.R_x, self.R_y, self.R_z = FuD.all_point(FuD.read_data()[0])
        self.P_1, self.P_2, self.P_name = Cout.Ports()
        self.NBI_X, self.NBI_Y, self.NBI_Z, self.NBI_uvec_X, self.NBI_uvec_Y, self.NBI_uvec_Z = Cout.NBI()
        self.Bget = calculus()


        # Create 3D surface
        self.surface = geo.create_surface(self.R_x, self.R_y, self.R_z)

        # Get intersections for ports and NBI
        self.new_P_1, *_ = geo.get_intersection_points(self.P_1, self.P_2, self.surface)
        self.new_NBI_start, self.new_NBI_end, *_ = geo.get_intersection_points_NBI(
            self.NBI_X, self.NBI_Y, self.NBI_Z, self.NBI_uvec_X, self.NBI_uvec_Y, self.NBI_uvec_Z, self.surface
        )
        self.valid_indx = []
        for i in range(0, 11):
            valid_indices = geo.pre_NBI_and_PORTS(i, self.new_P_1, self.new_NBI_start, self.new_NBI_end, self.surface)
            # Store the valid port names
            self.valid_indx.append(valid_indices)

    def port_for_nbi(self, NBI_index, angle, scale):
        P_1_for_NBI = self.new_P_1[:, self.valid_indx[NBI_index]]
        P_1_start_for_NBI = self.P_1[:, self.valid_indx[NBI_index]]
        Pname_for_NBI = [self.P_name[i] for i in self.valid_indx[NBI_index]]
        print(angle)
        valid_indices, extreme_points_1, extreme_points_2, *_ = geo.NBI_and_PORTS(
            P_1_start_for_NBI, NBI_index, P_1_for_NBI, self.new_NBI_start, self.new_NBI_end, self.surface, float(angle))
        valid_port_names = [Pname_for_NBI[i] for i in valid_indices]
        print(valid_port_names)
        return valid_indices, extreme_points_1, extreme_points_2, valid_port_names

    def data_already_input(self):
        Name_Ports = ['2_1_AEA', '2_1_AEM','2_1_AET'] 
        Name_NBI = ['NBI_7', 'NBI_8' ]
        Port_indices = [self.P_name.index(port) for port in Name_Ports if port in self.P_name]
        NBI_indices = [6,7]
        data = [[],[]]
        for i in range(len(NBI_indices)):
            NBI_index_i = NBI_indices[i]
            P_1_for_NBI_i = self.new_P_1[:, Port_indices]
            P_1_start_for_NBI = self.P_1[:, Port_indices]
            valid_indices, extreme_points_1, extreme_points_2, *_ = geo.NBI_and_PORTS(
            P_1_start_for_NBI, NBI_index_i, P_1_for_NBI_i, self.new_NBI_start, self.new_NBI_end, self.surface, float(90))
            data[i] = [Port_indices, np.array(extreme_points_1, dtype=np.float64), np.array(extreme_points_2, dtype=np.float64)]



        self.Bget.gets(data[0][1][1], data[0][2][1], 3)

        return data
    

class calculus():
    def __init__(self):
        pass




    def gets(self, point1, point2, scale):
      points = np.linspace(point1/100, point2/100, scale)
      os.chdir('J_0_test')
      mconf_config = {'B0': 2.525,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
      eq = mconf.Mconf_equilibrium('w7x-sc1.bc',mconf_config=mconf_config)
      for i in range(len(points)):
         B, vecB = eq.get_B(points[i])
         valueB = np.sqrt(vecB[0]**2 + vecB[1]**2 + vecB[2]**2)
         print(valueB)
      print(points)

        

if __name__ == "__main__":
    app = App()
    app.mainloop()

#files 
import Surface_rays as geo
import Surface_data as FuD
import NBI_Ports_data_input as Cout
import J_0_test.mconf.mconf as mconf
import Weight_Fuction.WF_FIDA as WF
import subprocess
import sys
import platform


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

libraries = [ "tkinter", "matplotlib","customtkinter", "numpy","scipy", "concurrent.futures","tqdm", "json", "jsonpickle"]
for library in libraries:
    try:
        globals()[library] = __import__(library)  
    except ImportError:
        print(f"Loading....: {library}")
        install(library)  


#library 
import tkinter as tk
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import numpy as np
import os 
from scipy.integrate import solve_ivp
from matplotlib import cm
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json
from tkinter import filedialog
import jsonpickle
from concurrent.futures import ThreadPoolExecutor

#====================================================================================================App_interface================================================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("W7-X cheking")
        self.geometry(f"{1600}x{900}")
        self.data_instance = Data()
        self.Bget = calculus()
        
        
        curr_directory = os.getcwd()
        print(curr_directory)
        # Global variable for slider value
        self.angle = int(90)
        self.scale = 10  
        self.delta_s = 0.05
        self.delta_J_0 = 0.01 
        self.current_graph = None
        self.all_results = []
        self.data_wf = []
        self.Name_Ports = ['2_1_AEA', '2_1_AEM','2_1_AET', '2_1_AEA', '2_1_AEM','2_1_AET']
        self.Name_NBI = ['NBI_7','NBI_7', 'NBI_7','NBI_8', 'NBI_8','NBI_8']
        self.results_folder = "Results"  
        self.conf_folder =os.path.join(curr_directory, "J_0_test")
        self.conf = 'w7x-sc1_ecrh_beta=0.02.bc'
        self.diagnostics = ['FIDA', 'FIDA', 'FIDA', 'FIDA', 'FIDA', 'FIDA']

  

        # Section 1: Sidebar
        self.create_sidebar()

        # Section 3: Additional Widgets
        self.create_additional_widgets()

        # Initialize default values
        self.nbi_optionmenu.set(self.nbi_options[0])
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.update_port_options(self.nbi_options[0])
    






    #------------------------------------------------------Interface---------------------------------------------------------------------------------------
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

    def create_additional_widgets(self):
        # create textbox
        self.textbox = ctk.CTkTextbox(self, width=30, font=("Arial", 14))  
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        welcome_message = (
            "Welcome to the program!\n\n"
            "Explanation of the naming convention:\n"
            "For '21AEM.S8':\n"
            "  - 21:  \t   Module 2, Submodule 1\n"
            "  - AEM: \t   Type of the port\n"
            "  - S:   \t   NBI (FIDA) \n"
            "  - C:   \t   Gyrotron (CTS) \n"
            "  - 8:   \t   NBI or C number\n\n"
            "Matrix Size setting:\n"
           "This adjusts the size of each matrix.\n\n"
        )

        self.textbox.insert("end", welcome_message, "\n\n")






        # Button to show graph
        self.show_graph_button = ctk.CTkButton(self, text="Add Port and build", command=lambda: self.dummy_function())
        self.show_graph_button.grid(row=1, column=1, padx=(20, 0), pady=(10, 0),sticky="nsew")

                # Button to show graph
        self.show_graph_button = ctk.CTkButton(self, text="Build", command=lambda: self.generate_and_show_graph())
        self.show_graph_button.grid(row=1, column=2, padx=(20, 0), pady=(10, 0), sticky="nsew")




        # create tabview with two sections
        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Section 1: "PORTS and NBI"
        self.tabview.add("Tools")
        self.tabview.tab("Tools").grid_columnconfigure(0, weight=1)

        # First CTkOptionMenu for NBI selection
        self.nbi_options = self.generate_nbi_options()
        self.nbi_optionmenu_label = ctk.CTkLabel(self.tabview.tab("Tools"), text="Select NBI or Gyrotron Launcher", anchor="w")
        self.nbi_optionmenu_label.grid(row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        self.nbi_optionmenu = ctk.CTkOptionMenu(self.tabview.tab("Tools"), values=self.nbi_options, command=self.update_port_options)
        self.nbi_optionmenu.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Second CTkOptionMenu for Port selection (based on NBI selection)
        self.port_options = []
        self.port_optionmenu_label = ctk.CTkLabel(self.tabview.tab("Tools"), text="Select Port", anchor="w")
        self.port_optionmenu_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.port_optionmenu = ctk.CTkOptionMenu(self.tabview.tab("Tools"), values=self.port_options)
        self.port_optionmenu.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")

        # Section 2: "Setting"
        self.tabview.add("Setting")
        self.tabview.tab("Setting").grid_columnconfigure(0, weight=1)

        # Slider in "Setting" section
        self.slider_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Viewing Angle:", anchor="w")
        self.slider_label.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        self.slider = ctk.CTkSlider(self.tabview.tab("Setting"), from_=30, to=90, command=self.slider_event)
        self.slider.set(self.angle)
        self.slider.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="w")

        # Label to display the current slider value
        self.value_label = ctk.CTkLabel(self.tabview.tab("Setting"), text=str(self.angle))
        self.value_label.grid(row=1, column=1, padx=(10, 20), pady=(10, 20), sticky="w")

        # Second slider for values 10 to 100 without intermediate values
        self.second_slider_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Matrix Size:", anchor="w")
        self.second_slider_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        self.second_slider = ctk.CTkSlider(self.tabview.tab("Setting"), from_=1, to=10, command=self.second_slider_event)
        self.second_slider.set(self.scale // 10)  # Set initial value based on second_slider_value
        self.second_slider.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="w")

        # Label for Delta s with Greek symbol
        self.delta_s_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="Δs:", anchor="w")
        self.delta_s_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")

        # Entry for Delta s value
        self.delta_s_entry = ctk.CTkEntry(self.tabview.tab("Setting"), width=100)
        self.delta_s_entry.insert(0, str(self.delta_s))  # Set initial value
        self.delta_s_entry.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="w")

        # Bind the entry to update value automatically when changed
        self.delta_s_entry.bind("<Return>", self.update_delta_s)


        # Label for Delta J with Greek symbol
        self.delta_J_label = ctk.CTkLabel(self.tabview.tab("Setting"), text="ΔJ0 %:", anchor="w")
        self.delta_J_label.grid(row=4, column=1, padx=20, pady=(10, 0), sticky="w")

        # Entry for Delta J value
        self.delta_J_entry = ctk.CTkEntry(self.tabview.tab("Setting"), width=100)
        self.delta_J_entry.insert(0, str(self.delta_J_0*100))  # Set initial value
        self.delta_J_entry.grid(row=5, column=1, padx=20, pady=(10, 20), sticky="w")

        # Bind the entry to update value automatically when changed
        self.delta_J_entry.bind("<Return>", self.update_delta_J)

        # Label to display the current value of the second slider
        self.second_value_label = ctk.CTkLabel(self.tabview.tab("Setting"), text=str(self.scale))
        self.second_value_label.grid(row=3, column=1, padx=(10, 20), pady=(10, 20), sticky="w")
        
        self.file_list_label = ctk.CTkLabel(self.tabview.tab("Tools"), text="Configuration", anchor="w")
        self.file_list_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.file_list = ctk.CTkOptionMenu(self.tabview.tab("Tools"), values=self.get_result_files_conf(), command=self.select_file_conf)
        self.file_list.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="w")


        self.file_list_label = ctk.CTkLabel(self.tabview.tab("Tools"), text="Results", anchor="w")
        self.file_list_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.file_list = ctk.CTkOptionMenu(self.tabview.tab("Tools"), values=self.get_result_files(), command=self.select_file)
        self.file_list.grid(row=7, column=0, padx=20, pady=(0, 20), sticky="w")


        self.save_button = ctk.CTkButton(self.tabview.tab("Tools"), text="Save data", command=self.save_results)
        self.save_button.grid(row=8, column=0, padx=20, pady=5, sticky="w")
        
        self.load_button = ctk.CTkButton(self.tabview.tab("Tools"), text="Load data", command=self.load_results)
        self.load_button.grid(row=9, column=0, padx=20, pady=5, sticky="w")

        # Canvas for displaying the graph
# Создаем холст для графика в колонке 3
        self.graph_canvas = ctk.CTkCanvas(self, bg="white")
        self.graph_canvas.grid(row=0, column=3, rowspan=999, padx=(20, 20), pady=(20, 20), sticky="nsew")

# Делаем третью колонку растягивающейся
        self.grid_columnconfigure(3, weight=1)  # 3-я колонка растягивается

# Делаем все строки растягивающимися
        for i in range(3):  # Или больше, если у тебя много строк
            self.grid_rowconfigure(i, weight=1)

    #--------------------------------------------------------------------------------------------------------------------------------------------- 







    #-------------------------------------------------------------Save data-------------------------------------------------------------------    
    def save_results(self):
        timestamp = datetime.now().strftime("%H%M%S")
        file_name = f"Results_{self.angle}_{self.scale}_{self.conf[:-3]}_{timestamp}.json"  
        file_path = os.path.join(self.results_folder, file_name)

        try:
            data_to_save = {
            'scale': self.scale,
            'angle': self.angle,
            'results': self.all_results,
            'Ports': self.Name_Ports,
            'NBI':self.Name_NBI,
            'conf': self.conf
        }
            with open(file_path, 'w') as f:
                f.write(jsonpickle.dumps(data_to_save))

            self.textbox.insert("end", f"[{timestamp}]: File was saved: {file_path}\n")
            self.file_list.configure(values=self.get_result_files())  
        except Exception as e:
            self.textbox.insert("end", f"[{timestamp}]: Error during saving file: {e}\n")
            print(f"Error during saving file: {e}")

        
    def load_results(self):
        file_path = filedialog.askopenfilename(initialdir=self.results_folder, filetypes=[("JSON files", "*.json")])
        if file_path:
            self.load_from_file(file_path)
    
    def select_file(self, file_name):
        file_path = os.path.join(self.results_folder, file_name)
        self.load_from_file(file_path)
    
    def load_from_file(self, file_path):
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            with open(file_path, 'r') as f:
                #content = f.read()
                #print(len(content))
                data = jsonpickle.loads(f.read())
            self.scale = data['scale']
            self.angle = data['angle']
            self.all_results = data['results']
            self.Name_Ports = data['Ports']
            self.Name_NBI = data['NBI']
            self.conf = data['conf']
            self.data_wf = self.all_results[10][0]
            new_names = self.port_name()
            self.textbox.insert("end", f"\n[{timestamp}]: Loaded Data:\nMatrix size: {self.scale}\nViewing Angle: {self.angle}\nConfiguration: {self.conf} \nPorts: {', '.join(new_names)} \n  ")
            #Diagnostics: {', '.join(self.diagnostics)}\n
        except Exception as e:
            self.textbox.insert("end", f"\n[{timestamp}]: Error during loading file: {e}\n")
            

    def port_name(self):
        new_names = []
        for i in range(len(self.Name_Ports)):
            selected_nbi = self.Name_NBI[i]
            selected_port = self.Name_Ports[i]
            if selected_nbi[0] == 'N':
                name = 'S'
            else:
                name = 'C'
            new_name = f'{selected_port[0]}{selected_port[2]}{selected_port[4:]}.{name}{selected_nbi[4]}'
            new_names.append(new_name)
        return new_names
    
    def get_result_files(self):
        if os.path.exists(self.results_folder):
         return [f for f in os.listdir(self.results_folder) if f.endswith(".json")]
        else:
          print(f"Directory {self.results_folder} not found.")
          return []
    
    def get_result_files_conf(self):
        files = [f for f in os.listdir(self.conf_folder) if f.endswith((".bc", ".txt", ".nc"))]

        if self.conf in files:
         files.remove(self.conf)
         files.insert(0, self.conf)
        return files
    
    def select_file_conf(self, file_name):
        self.conf = file_name
        print(self.conf)
    #------------------------------------------------------------------------------------------------------------------------------------------





     
    #-----------------------------------------------Pre_calculate------------------------------------------------------------------------------
    def pre_calculate(self):
        if len(self.all_results) ==0:
         Result_array = self.data_instance.data_already_input(self.scale,  self.Name_Ports,   self.Name_NBI, self.angle, self.conf)
         self.all_results = Result_array
         self.data_wf = self.all_results[10][0]
        else:
            self.all_results = [row[:6] for row in self.all_results]

        time = datetime.now().strftime("%H:%M:%S")
        self.textbox.insert("end", f"\n [{time}]: Old data ready \n\n ")
    #---------------------------------------------------------------------------------------------------------------------------------------------
        
    


        
    #--------------------------------------------------------------Setting functions---------------------------------------------------------------
    def second_slider_event(self, value): 
        self.scale = int(value) * 10  
        self.second_value_label.configure(text=str(self.scale))


    def slider_event(self, value):  #about angle 
        self.angle = int(value)
        self.value_label.configure(text=str(self.angle))


    def update_delta_s(self, event):
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            new_value = float(self.delta_s_entry.get())
            self.delta_s = new_value
            print(f"Delta s updated to: {self.delta_s}")

            self.textbox.insert("end", f"\n[{timestamp}]: Δs updated to: {self.delta_s}\n")
        except ValueError:
            self.textbox.insert("end", f"\n[{timestamp}]: Invalid input. Please enter a numerical value.\n")

    def update_delta_J(self, event):
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            new_value = float(self.delta_J_entry.get())
            self.delta_J_0 = new_value/100
            print(f"Delta J updated to: {self.delta_J_0}")


            self.textbox.insert("end", f"\n[{timestamp}]: ΔJ0 updated to: {self.delta_J_0}\n")
        except ValueError:
            self.textbox.insert("end", f"\n[{timestamp}]: Invalid input. Please enter a numerical value.\n")
    #-----------------------------------------------------------------------------------------------------------------------------------------------------------


    #------------------------------------------------------------Tools function----------------------------------------------------------------------------
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
     self.port_optionmenu.configure(values=self.port_options)  
     if self.port_options:
            self.port_optionmenu.set(self.port_options[0])

    def generate_nbi_options(self):
      return [f"NBI_{i}" if i <= 8 else f"CTS_{i-8}" for i in range(1, 12)]
    #-----------------------------------------------------------------------------------------------------------------------------------------------------
 



    #------------------------------------------------------FriendlyUsefull-----------------------------------------------------------------------------------
    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)
    #-------------------------------------------------------------------------------------------------------------------------------------------------------

    



    #---------------------------------------Add new matrxi to grid ----------------------------------------------------
    def create_result_array_for_port(self, selected_nbi, selected_port):
        data = self.data_instance.data_nbi_ports(selected_nbi, selected_port, self.angle, self.scale, self.conf)

        for i in range(len(data)):
            self.all_results[i].append(data[i])

        reuslt = self.data_instance.process_data_for_point(len(self.all_results[0])-1, self.all_results)
        self.data_wf.append(reuslt)
    #------------------------------------------------------------------------------------------------------------------






    #---------------------------------------------------------------------Graph----------------------------------------------------
    def generate_and_show_graph(self):
        if self.current_graph:
            self.current_graph.get_tk_widget().destroy()

        self.draw_graph_on_canvas(self.data_wf)

    def dummy_function(self):
        #User 
        selected_nbi = self.nbi_optionmenu.get()
        selected_port = self.port_optionmenu.get()
        
        #Time
        timestamp = datetime.now().strftime("%H:%M:%S")
          
        #Message  
        message = f"\n[{timestamp}]: Selected Port:    {selected_port}\nSelected:     {selected_nbi}\n"
        self.textbox.insert("end", message)

        #Data
        self.Name_NBI.append(selected_nbi)
        self.Name_Ports.append(selected_port)
        self.create_result_array_for_port(selected_nbi, selected_port)

        self.generate_and_show_graph()
        
        
    def draw_graph_on_canvas(self, Result_for_NBI_Port):
        num_arrays = len(Result_for_NBI_Port)
        color = np.array([])
        Matr= np.empty((num_arrays, num_arrays), dtype=object)
            
        # Create a matplotlib figure
        fig, axs = plt.subplots(num_arrays, num_arrays, figsize=(10, 10))

        for i in range(num_arrays):
            for j in range(num_arrays):
                 MATRIX = self.sum(Result_for_NBI_Port[i], Result_for_NBI_Port[j], i, j)
                 MATRIX= np.transpose(MATRIX)
                 filtered_values = MATRIX[MATRIX != -np.inf]
                 min_value = np.min(filtered_values) if filtered_values.size > 0 else -9
                 min_value = 3.0
                 color = np.append(color, min_value)
                 Matr[i, j]  = MATRIX
        for i in range(num_arrays):
            for j in range(num_arrays):

                One_Matr = Matr[i, j] 
                max_value =np.max(MATRIX)
                max_value = 4.0
                im = axs[i, j].imshow(One_Matr, cmap='jet', origin='upper', aspect='auto', vmin=min_value, vmax=max_value)
                #gist_ncar
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0)
        
        
        # Add colorbar to the last subplot
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
        
        plt.colorbar(im, cax=cax)


        for i in range(num_arrays):
            if num_arrays>=11:
                fonts = 6
            else:
                fonts = 9

            selected_nbi = self.all_results[0][i]
            selected_port = self.Name_Ports[i]
            if selected_nbi[0] == 'N':
               name = 'S'
            else:
               name = 'C'
            axs[num_arrays-1, i].set_xlabel(f'{selected_port[0]}{selected_port[2]}{selected_port[4:]}.{name}{selected_nbi[4]}', fontsize=fonts)
            axs[i, 0].set_ylabel(f'{selected_port[0]}{selected_port[2]}{selected_port[4:]}.{name}{selected_nbi[4]}', fontsize=fonts)

        file_name = f"Results_{self.angle}_{self.scale}_{self.conf[:-3]}.png"  
        fig.savefig(f"Results/{file_name}", dpi=2000, bbox_inches='tight')

        #plt.title('FIDA')
        plt.close('all')

        # Embed the matplotlib figure in the Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.graph_canvas)
        canvas_widget = canvas.get_tk_widget()
        #canvas_widget.grid(row=0, column=0, padx=(10,10), pady=(10, 0), sticky="nsew")
        canvas_widget.pack(fill='both', expand=True)

        # Update the current graph reference
        self.current_graph = canvas


    def sum(self, array_1, array_2, first_nbi_index, second_nbi_index):
     MATRIX = np.zeros((len(array_1), len(array_2)))

     
     x_ev = np.linspace(10, 100, 300)
     y_ev = np.linspace(-100, 100, 300) / 2.5
     x_ev, y_ev = np.meshgrid(x_ev, y_ev)

     ratio = np.abs(x_ev / y_ev)

     B_value_1 = self.all_results[3][first_nbi_index]
     B_value_2 = self.all_results[3][second_nbi_index]
     B_max_array_1 = self.all_results[8][first_nbi_index]
     B_max_array_2 = self.all_results[8][second_nbi_index]
     s_1_array = self.all_results[7][first_nbi_index]
     s_2_array = self.all_results[7][second_nbi_index]
     J_0_array_1 = np.array(self.all_results[9][first_nbi_index])  
     J_0_array_2 = np.array(self.all_results[9][second_nbi_index])  

     for i in range(len(array_1)):
        for j in range(len(array_2)):
            B_max_i = B_max_array_1[i]
            B_max_j = B_max_array_2[j]
            s_1_i = s_1_array[i]
            s_2_j = s_2_array[j]

            mask_i = (ratio > B_max_i)
            mask_j = (ratio > B_max_j)

            mask_i_val = (ratio> B_value_1[i])
            mask_j_val = (ratio> B_value_2[j])
            
            x_idx = i % x_ev.shape[0]  
            y_idx = j % y_ev.shape[1]  
            
            J_0_i = J_0_array_1[x_idx, y_idx]  
            J_0_j = J_0_array_2[x_idx, y_idx]  

            #free particles 
            both_above_B_mask = mask_i & mask_j

            #not free
            both_below_B_mask = np.logical_not(mask_i) & np.logical_not(mask_j) & mask_i_val & mask_j_val

            #or or 
            cross_check_mask = (mask_i & np.logical_not(mask_j)) | (np.logical_not(mask_i) & mask_j)

            #==================================free=================================================
            ratio_mask_s = (np.abs(np.abs(s_1_i / s_2_j)-1) <=  self.delta_s)
            equal_s_mask = ratio_mask_s 
            #==================================NOT free=================================================
            def relative_difference(a, b):
                numerator = np.abs(a - b)

                denominator = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
                return numerator / denominator


            J_0_check_mask = relative_difference(J_0_i, J_0_j) < self.delta_J_0
            #==================================forbidden======================================
            mask_no_accept =  np.logical_not(mask_i_val) | np.logical_not(mask_j_val) 

            #free diff
            mask_condition_1 = both_above_B_mask  & np.logical_not(equal_s_mask)
            #free same
            mask_condition_2 = both_above_B_mask & equal_s_mask
            #or or 
            mask_condition_4 = cross_check_mask
            #not free same 
            mask_condition_5 = both_below_B_mask & J_0_check_mask
            #not free diff 
            mask_condition_6 = both_below_B_mask & np.logical_not(J_0_check_mask)
            #f
            mask_condition_7 = mask_no_accept
            #  
            mask_condition_8 = (s_1_i > 1) | (s_2_j > 1)
            #results
            product = array_1[i] * array_2[j]
            #mask
            product[mask_condition_1] = 0
            product[mask_condition_4] = 0
            product[mask_condition_6] = 0
            product[mask_condition_7] = 0
            product[mask_condition_8] = 0

            sum_product = np.sum(product)
            element = np.log10(np.where(sum_product > 0, sum_product, 1e-8))
            MATRIX[i, j] = np.where(element>-8, element, -np.inf)

     return MATRIX




class Data:
        #====================================================================DATA TYPE ====================================================================================================
        #data_B: [Name NBI; Name Port; Points on NBI; Mag field in this points; Angle between linesight and vec NBI; vec Mag field in points on NBI; angle between vec linesi and magfield]
        #data_B[0]: Name NBI
        #data_B[1]: Name Port
        #data_B[2]: Points on NBI
        #data_B[3]: Mag field in this points
        #data_B[4]: Angle between linesight and vec NBI
        #data_B[5]: vec Mag field in points on NBI
        #data_B[6]: angle between vec linesi and magfield
        #data_B[7]: S
        #data_B[8]: B_max 
        #data_B[9]: J_0
        #data_B[10]: WF
        #==================================================================================================================================================================================



    def __init__(self):
        self.R_x, self.R_y, self.R_z = FuD.all_point(FuD.read_data()[0])
        self.P_1, self.P_2, self.P_name = Cout.Ports()
        self.NBI_X, self.NBI_Y, self.NBI_Z, self.NBI_uvec_X, self.NBI_uvec_Y, self.NBI_uvec_Z = Cout.NBI()
        self.Bget = calculus()



        # ===================================================================================================Create 3D surface=========================================================
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
        #============================================================================================================================================================================

    def port_for_nbi(self, NBI_index, angle, scale):
        P_1_for_NBI = self.new_P_1[:, self.valid_indx[NBI_index]]
        P_1_start_for_NBI = self.P_1[:, self.valid_indx[NBI_index]]
        Pname_for_NBI = [self.P_name[i] for i in self.valid_indx[NBI_index]]
        valid_indices, extreme_points_1, extreme_points_2, *_ = geo.NBI_and_PORTS(
            P_1_start_for_NBI, NBI_index, P_1_for_NBI, self.new_NBI_start, self.new_NBI_end, self.surface, float(angle))
        valid_port_names = [Pname_for_NBI[i] for i in valid_indices]
        return valid_indices, extreme_points_1, extreme_points_2, valid_port_names
    

    def data_already_input(self, scale, Name_Ports, Name_NBI, angle, config):

        data_B = [[] for _ in range(11)]
        for i in range(len(Name_Ports)):
           data_i = self.data_nbi_ports(Name_NBI[i], Name_Ports[i], angle, scale, config)
           
           for i in range(len(data_i)):
               data_B[i].append(data_i[i])
           print(data_B[0])

        WF_array = self.create_result_array_for_port_old(data_B)
        data_B[10].append(WF_array)

        return data_B
    


    def data_nbi_ports(self, nbi, port, angle, scale, config):
        index = self.P_name.index(port)
        P_1_start = [self.P_1[0][index], self.P_1[1][index], self.P_1[2][index]]
        P_2_end = [self.new_P_1[0][index], self.new_P_1[1][index], self.new_P_1[2][index]]
        
        index_NBI = int(nbi.split('_')[-1])-1
        if nbi.startswith("CTS"):
            index_NBI += 8

        valid_indices, extreme_points_1, extreme_points_2, *_ = geo.NBI_and_PORTS(
            P_1_start, index_NBI, P_2_end, self.new_NBI_start, self.new_NBI_end, self.surface, float(angle))

        points, B_array, B_vec_array, S_array, B_max_array= self.Bget.gets(np.array(extreme_points_1[0], dtype=np.float64), np.array(extreme_points_2[0], dtype=np.float64), scale, config)

        angles, angles_vec_B, J_0_array=[],[], []
        for j in range(len(points)):
                angle = geo.check_segment_angle(P_1_start, P_2_end, points[j]*100)
                angles.append(angle)
                vector_AB = np.array(P_2_end) - np.array(P_1_start)  
                angle_B = geo.check_angle_2_vec(vector_AB/100, B_vec_array[j])
                angles_vec_B.append(angle_B)
        J_0_array = self.Bget.J_0_calculate(points, config)

        return [nbi, port, points, B_array, angles, B_vec_array, angles_vec_B,S_array, B_max_array, J_0_array] 


    def create_result_array_for_port_old(self, data_B_c):
        WF_array = []

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.process_data_for_point, range(len(data_B_c[1])), [data_B_c] * len(data_B_c[0]))))

        for result_for_i in results:
            WF_array.append(result_for_i)

    
        return results

    def process_data_for_point(self,i, data_B_c):

        index_nbi = int(data_B_c[0][i].split('_')[-1]) - 1
        result_for_j = []

        for j in range(len(data_B_c[3][i])):
            x_ev = np.linspace(10, 100, 300)
            y_ev = np.linspace(-100, 100, 300) / 2.5

            if index_nbi < 8:
               result = WF.weight_Function(data_B_c[4][i][j], data_B_c[3][i][j], x_ev, y_ev)
            else:
                result = WF.CTS_wf(data_B_c[6][i][j], data_B_c[3][i][j], x_ev, y_ev)
            result_for_j.append(result) 
        return result_for_j




class calculus():
    def __init__(self):
         lll=0

    def gets(self, point1, point2, scale, config):
      point1, point2 = point1 / 100, point2 / 100
      points = np.linspace(point1, point2, scale)

      previous_directory = os.getcwd()
      os.chdir('J_0_test')


      mconf_config = {'B0': 2.911,
                'B0_angle': 0.0,
                'accuracy': 1e-10, 
                'truncation': 1e-10} 
      eq = mconf.Mconf_equilibrium(config ,mconf_config=mconf_config)

      
      S_array = []
      for i in range(len(points)):
         S, vecB = eq.get_B(points[i])
         S_array.append(S)
      start_indices, end_indices = self.find_transitions(S_array)
      start_point = points[start_indices[0]].reshape(3,)
      end_point = points[end_indices[0]].reshape(3,)


      points = np.linspace(start_point, end_point, scale)

      B_array, B_vec_array, S_array, B_max_array= [], [], [], []
      for i in range(len(points)):
         S, vecB = eq.get_B(points[i])
         B_max = eq.get_Bmax(S)
         valueB = np.sqrt(vecB[0]**2 + vecB[1]**2 + vecB[2]**2)
         S_array.append(S)
         B_array.append(valueB)
         B_vec_array.append(vecB)
         S_array.append(S)
         B_max_array.append(B_max)

      os.chdir(previous_directory)
      return points, B_array, B_vec_array, S_array, B_max_array
    

    def find_transitions(self, arr):
        start_indices = []
        end_indices = []
    
        for i in range(len(arr) - 1):
            if arr[i] >= 1 and arr[i+1] < 1:   
                start_indices.append(i+1)
            elif arr[i] < 1 and arr[i+1] >= 1: 
                end_indices.append(i+1)
    
        return start_indices, end_indices







    #------------------------------------------------------------------------------J_0 calc-------------------------------------------------------------------------------------------------
    def J_0_calculate(self, points,config):
     points = np.array(points, dtype=np.float64)
     previous_directory = os.getcwd()
     os.chdir('J_0_test')
    
     with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(self.calculate_J_0_for_point, points, [config] * len(points)), total=len(points)))
     print(config)
     os.chdir(previous_directory)
     return results

    def calculate_J_0_for_point(self, point, config):
      #data type
      point = np.array(point, dtype=np.float64)

      #config
      mconf_config = {'B0': 2.911,
                'B0_angle': 0.0,
                'accuracy': 1e-10, #accuracy of magnetic to cartesian coordinat transformation
                'truncation': 1e-10} #trancation of mn harmonics
      eq = mconf.Mconf_equilibrium(config,mconf_config=mconf_config)
      

      #constant
      s0, vecB = eq.get_B(point)
      B_value = np.linalg.norm(vecB)
      B_max_point = eq.get_Bmax(s0)
      L = 300 
      N = 2000
      E_values = np.linspace(10, 100, 300) * (1.6 * 10**(-19)) * 10**3  
      mu_values = np.linspace(-100, 100, 300)/2.5 * (1.6 * 10**(-19)) * 10**3 

      #Solve eq
      def solve_differential(point, L, N, rhs_B):
        sol = solve_ivp(rhs_B, [0, L], point,method='RK45', max_step=L / N,atol=1e-6, dense_output=True)
        s_sol, B_sol = eq.get_s_B_T(sol.y[0],sol.y[1], sol.y[2])
        magB = np.linalg.norm(B_sol, axis=1)
        path = np.zeros(sol.y.shape[1])
        path = np.cumsum(np.sqrt(np.sum(np.diff(sol.y,axis=-1)**2, axis=0)))
        path = np.insert(path, 0, 0) 
        return magB, path

      
      rhs_B_forward = lambda l, y: eq.get_B(y)[1]/ np.linalg.norm(eq.get_B(y)[1])
      rhs_B_backward = lambda l, y: -eq.get_B(y)[1]/ np.linalg.norm(eq.get_B(y)[1])

      forward_magB, forward_path = solve_differential(point, L, N, rhs_B_forward)
      backward_magB, backward_path = solve_differential(point, L, N, rhs_B_backward)

    
      #Arrays
      E_values, mu_values = np.meshgrid(E_values, mu_values)
      J_0_map = np.zeros(E_values.shape)

      for i in range(E_values.shape[0]):
          for j in range(mu_values.shape[1]):
              
              E = E_values[i, j]
              mu = mu_values[i, j]
              B_max_particle = np.abs(E / mu)
              
              
              if B_max_particle<B_max_point and B_max_particle>B_value and s0<1:
               
               #-------------------Integral----------------------------
               def compute_integrals(magB, path, mu, B_max_particle):
                 mask = magB <= B_max_particle
                 idx_limit = np.argmax(~mask) if np.any(~mask) else len(magB)
                 magB_limited, path_limited = magB[:idx_limit], path[:idx_limit]
                 integrand = np.sqrt(2 * np.abs(mu) * (B_max_particle - magB_limited))
                 return integrand, path_limited
    

               forward_integrand, forward_path_limited = compute_integrals(forward_magB, forward_path, mu, B_max_particle)
               backward_integrand, backward_path_limited = compute_integrals(backward_magB, backward_path, mu, B_max_particle)
              

               complete_path = np.concatenate([
                backward_path_limited[::-1],  
                forward_path_limited,         
                forward_path_limited[::-1],   
                backward_path_limited         
                ])

               complete_integrand = np.concatenate([
                 backward_integrand[::-1],    
                 forward_integrand,           
                 forward_integrand[::-1],     
                 backward_integrand          
                 ])
               
               J_0_map[i, j] = self.trapezoidal_integral(complete_integrand, complete_path)*1e7
              else:
               J_0_map[i, j] = 0.0     

      return J_0_map
    
    def trapezoidal_integral(self, f, s):
      ds = np.abs(np.diff(s))
      avg_f = (f[:-1] + f[1:]) / 2
      segment_integrals = ds * avg_f
      integral = np.sum(segment_integrals)
      return integral
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    app = App()
    app.mainloop()

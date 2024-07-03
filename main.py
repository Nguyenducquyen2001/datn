import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np
import time 
import os
import matplotlib.pyplot as plt
import torch
from ai import ensemble_model

def main():
    # import recive_save as dt



    # model_path = 'F:\hk2_nam4\đatn\web\model'
    # model_filename = 'best_model.pt'

    # # Load the model
    # model = LSTMModel(1,10,2)
    # model.load_state_dict(torch.load(os.path.join(model_path, model_filename)))
    # # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model.to(device)

    # # Set the model to evaluation mode
    # model.eval()

    # # Set the page configuration


    st.markdown("""
        <style>
        body {
            background-color: #FFFFCC; 
            background-size: 800px; /* Chiều rộng nền là 800px */
            background-repeat: no-repeat; /* Không lặp lại nền */
            
        }
        .stApp {
            background: none;
            padding: 0;
        }
        .block-container {
            background-color: white;  /* White background for the main content area */
            padding: 20px;
            # border-radius: 10px;
            box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.7);  /* Shadow to make the main content area stand out */
            max-width: 800px;  /* Max width for the main content area */
            margin: 60px auto 0 auto;
            position: relative; /* Ensure relative positioning for child elements */
            overflow-y: auto; /* Add vertical scrollbar if content exceeds max height */
        }
        </style>
        """, unsafe_allow_html=True)


    with st.sidebar:
        selected = option_menu(
            menu_title = "",
            options = ["Header","Data","Prediction"],
            icons = ["house", "table", "bar-chart"],
            # icons = ["house","info-circle", "database", "graph-up", "pencil-square"],
            default_index = 0,
            styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "blue", "font-size": "24px"},
                    "nav-link": {
                        "font-size": "24px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "#DDE0E6"},
                }
        )       


    # Xử lý tùy chọn được chọn
    if selected == "Header":



        col1,  col2=st.columns([5,1])
        with col1:
            st.image("img/trường spkt.png", width=550)
        with col2:
            st.image("img/LOGO_KHOA_CKM.png", width=92)
            
        st.markdown("""
        <div style="height: 50px;"></div>
        """, unsafe_allow_html=True)  
        
        st.markdown("""
        <div style="background-color: #000080; padding: 20px; border-radius: 10px;">
            <div style="text-align: center;">
                <p style="font-size: 30px; font-family:Times New Roman; padding-bottom: 20px; color: white;">
                    ĐỒ ÁN TỐT NGHIỆP
                </p>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 21px; font-family:Times New Roman; color: white;">
                    NGHIÊN CỨU THIẾT LẬP MÔ HÌNH ĐO RUNG ĐỘNG CỦA CÁN DAO TIỆN VÀ ỨNG DỤNG HỌC MÁY ĐỂ ĐÁNH GIÁ ĐỘ ỔN ĐỊNH CỦA DAO
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="height: 20px;"></div>
        """, unsafe_allow_html=True) 
    
        col_img, col_name = st.columns([2, 4])

        #Hình ảnh của HVHD (hoặc hình ảnh khác)
        with col_img:
            st.markdown("""
            <div style="height: 20px;"></div>
            """, unsafe_allow_html=True) 
            st.image("F:/hk2_nam4/đatn/web/img/2524.jpg_wh860.jpg", use_column_width="auto")
            
        #Thông tin về HVHD sang nửa bên phải
        with col_name:
            st.markdown("""
                <div style="padding: 20px; border-radius: 10px;">
                    <h2 style='font-size: 16px; font-family:Times New Roman;'>GVHD: PSG. TS. Đỗ Thành Trung</h2>   
                    <div class="row" style="display: flex;">
                        <div style="width: 60%; ">
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-top: -25px;'>SVTH:</h2>
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-left: 60px; margin-top: -25px;'>Lê Toàn Phát</h2>
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-left: 60px; margin-top: -25px;'>Nguyễn Tấn Phát</h2>
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-left: 60px; margin-top: -25px;'>Nguyễn Đức Quyền</h2>
                        </div>
                        <div style="width: 40%; padding: 10px;">
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-top: -35px;'>MSSV:</h2>
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-left: 40px; margin-top: -25px;'>20146510</h2>
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-left: 40px; margin-top: -25px;'>20146511</h2>
                            <h2 style='font-size: 16px; font-family:Times New Roman; margin-left: 40px; margin-top: -25px;'>20146148</h2>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        
        st.markdown("""
        <div style="height: 55px;"></div>
        """, unsafe_allow_html=True)   


    if selected == "Data":
        
        def main():
            
            show_plt_F_a=160
            show_plt_S=88000
            # start_index = 7
            # end_index =  start_index+1 

            plot_container1 = st.empty() 
            plot_container2 = st.empty() 
            plot_container3 = st.empty() 
            while True:
                #cập nhật gia tốc 
                csv_F_a ='gui/data2.csv'     
                data_F_a = pd.read_csv(csv_F_a, header=None)
                data_F_a[0] = (data_F_a[0] * 3.3 * 1000) / (2**24 * 128)
                data_F_a[1] = (data_F_a[1] * 3.3 * 1000) / (2**24 * 128)
            
                #cập nhật âm thanh
                csv_S='gui/data3.csv' 
                data_S = pd.read_csv(csv_S).values.reshape(-1)/3277
                # data_S.columns = ['sound']  
                
                # vẽ gia tốc
                fig, ax = plt.subplots(figsize=(12, 4))
                # ax.plot(data_F_a.index[start_index*160:end_index*160], data_F_a[start_index*160:end_index*160][data_F_a.columns[2]], label='ax')
                # ax.plot(data_F_a.index[start_index*160:end_index*160], data_F_a[start_index*160:end_index*160][data_F_a.columns[3]], label='ay')
                # ax.plot(data_F_a.index[start_index*160:end_index*160], data_F_a[start_index*160:end_index*160][data_F_a.columns[4]], label='az')
                ax.plot(data_F_a.index[-show_plt_F_a:], data_F_a[-show_plt_F_a:][data_F_a.columns[2]], label='ax')
                ax.plot(data_F_a.index[-show_plt_F_a:], data_F_a[-show_plt_F_a:][data_F_a.columns[3]], label='ay')
                ax.plot(data_F_a.index[-show_plt_F_a:], data_F_a[-show_plt_F_a:][data_F_a.columns[4]], label='az')
                ax.set_xlabel('SAMPLE')
                ax.set_ylabel('ACCELERATION')
                ax.set_ylim(-0.25, 0.25)
                ax.legend()
                plot_container1.pyplot(fig)
                
                # vẽ lực 
                fig, ax1 = plt.subplots(figsize=(12, 4))
                # ax1.plot(data_F_a.index[start_index*160:end_index*160], data_F_a[start_index*160:end_index*160][data_F_a.columns[0]], label='Dz')
                # ax1.plot(data_F_a.index[start_index*160:end_index*160], data_F_a[start_index*160:end_index*160][data_F_a.columns[1]], label='Dx')
                ax1.plot(data_F_a.index[-show_plt_F_a:], data_F_a[-show_plt_F_a:][data_F_a.columns[0]], label='Dz')
                ax1.plot(data_F_a.index[-show_plt_F_a:], data_F_a[-show_plt_F_a:][data_F_a.columns[1]], label='Dx')
                ax1.set_xlabel('SAMPLE')
                ax1.set_ylabel('DEFOMATION')
                ax.set_ylim(-1, 1)
                ax1.legend()
                plot_container2.pyplot(fig)
                
                # vẽ âm thanh 
                fig, ax2 = plt.subplots(figsize=(12, 4))
                ax2.plot(range(len(data_S) - 88000,len(data_S)), data_S[-show_plt_S:], label='sound')
                # ax2.plot(range(start_index*176000,end_index*176000), data_S[-show_plt_S:], label='sound')
                ax2.set_xlabel('SAMPLE')
                ax2.set_ylabel('SOUND')
                ax2.legend()
                ax.set_ylim(-1, 1)
                plot_container3.pyplot(fig)
                    
                time.sleep(0.1)

        if __name__ == '__main__':
            main()
    if selected == "Prediction": 
        
        st.markdown("""
            <div style="text-align: center;">
                <p style="font-size: 30px; font-family:Times New Roman; color: blue;">
                    CURRENT SYSTEM OPERATION        
                </p>
            </div>
            """, unsafe_allow_html=True)
        # df_acc_sg = pd.read_csv("./gui/data2.csv", header = None)
        # df_audio = pd.read_csv("./gui/data3.csv", header = None)
        # if len(df_acc_sg) > 160 and len(df_audio) > 172:
        df_acc_sg = pd.read_csv("./gui/data2.csv", header = None)
        df_audio = pd.read_csv("./gui/data3.csv", header = None)
        results = []
        plot_container5 = st.empty() 
        plot_container6 = st.empty()
        plot_container7 = st.empty() 
        
        st.markdown("""
        <div style="height: 100px;"></div>
        """, unsafe_allow_html=True)
         
        if len(df_acc_sg) > 160 and len(df_audio) > 172:
            while True:
                df_acc_sg = pd.read_csv("./gui/data2.csv", header = None)
                df_audio = pd.read_csv("./gui/data3.csv", header = None)
                proba0, result = ensemble_model(df_acc_sg[-160:], df_audio[-172:])
                proba0=proba0*100
                results.append(result)
                if len(results)> 1:                              
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(range(len(results) - min(len(results), 10), len(results)), results[-10:], label='Result')
                    ax.set_xlabel('SAMPLE')
                    ax.set_ylabel('RESULT')
                    ax.legend()
                    plot_container5.pyplot(fig)
                    if result == 0:
                        plot_container6.write("STABLE TURNING PROCESS")
                    else:
                        plot_container6.warning("UNSTABLE TURNING PROCESS")
                    plot_container7.write(f"The percentage of stable systems is {proba0:.2f}%")   
                    
    # if selected == "chỉnh":  
    #     st.markdown("""
    #         <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px;">
    #             <h2 style='font-size: 30px;text-align: center;'>GIÁO VIÊN HƯỚNG DẪN </h2>
    #             <div style="display: flex; justify-content: space-between; padding-left: 55px;" class="col-6">
    #                 <div class="col-6">
    #                     <h2 style='font-size: 16px;'>HVHD1</h2>
    #                     <ul style='list-style-type: none;'>
    #                         <li>Họ và tên: Đỗ Thành Trung</li>
    #                         <li>Chức vụ: PGS Tiến Sĩ</li>
    #                         <li>SĐT: 0989881588</li>
    #                     </ul>
    #                 </div>
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)
        
    #     st.markdown("""
    #     <div style="height: 20px;"></div>
    #     """, unsafe_allow_html=True)    
        
    #     st.markdown("""
    #         <div style="background-color: #f0f8ff; padding: 10px; border-radius: 10px;">
    #             <h2 style='font-size: 30px; text-align: center;'>SINH VIÊN THỰC HIỆN</h2>
    #             <div class="row">
    #                 <div style="display: flex; justify-content: space-around;" class="col-6">
    #                     <div>
    #                         <h2 style='font-size: 16px;'>SVTH1</h2>
    #                         <ul style="list-style-type: none;">
    #                             <li>Họ và tên: Nguyễn Đức Quyền</li>
    #                             <li>MSSV: 20146148</li>
    #                             <li>Ngành: CNKT Cơ Điện Tử</li>
    #                             <li>SĐT: (+84)377247668</li>
    #                         </ul>
    #                     </div>
    #                     <div>
    #                         <h2 style='font-size: 16px;'>SVTH2</h2>
    #                         <ul style="list-style-type: none;">
    #                             <li>Họ và tên: Nguyễn Tấn Phát</li>
    #                             <li>MSSV: 20146511</li>
    #                             <li>Ngành: CNKT Cơ Điện Tử</li>
    #                             <li>SĐT: (+84)784174102</li>
    #                         </ul>
    #                     </div>
    #                 </div>
    #                 <div style="display: flex; justify-content: space-between; padding-left: 55px;" class="col-6">
    #                     <div>
    #                         <h2 style='font-size: 16px;'>SVTH3</h2>
    #                         <ul style="list-style-type: none;">
    #                             <li>Họ và tên: Lê Toàn Phát</li>
    #                             <li>MSSV: 20146510</li>
    #                             <li>Ngành: CNKT Cơ Điện Tử</li>
    #                             <li>SĐT: (+84)385548477</li>
    #                         </ul>
    #                     </div>
    #                 </div>
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)

main()
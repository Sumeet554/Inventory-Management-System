import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CRMDataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = self.df.shape
        
    def clean_data(self):
        """
        Clean the dataset by handling missing values, duplicates, and data type conversions
        """
        # Convert date columns to datetime
        date_columns = [
            'first_order_date', 'last_order_date',
            'last_order_date_online', 'last_order_date_offline'
        ]
        
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            
        # Handle missing values
        numeric_columns = [
            'order_num_total_ever_online', 'order_num_total_ever_offline',
            'customer_value_total_ever_offline', 'customer_value_total_ever_online'
        ]
        
        # Fill numeric missing values with 0
        self.df[numeric_columns] = self.df[numeric_columns].fillna(0)
        
        # Clean category interests
        if 'interested_in_categories_12' in self.df.columns:
            self.df['interested_in_categories_12'] = self.df['interested_in_categories_12'].fillna('unknown')
            self.df['interested_in_categories_12'] = self.df['interested_in_categories_12'].str.strip()
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['master_id'], keep='first')
        
        # Log cleaning results
        cleaning_results = {
            'original_rows': self.original_shape[0],
            'cleaned_rows': self.df.shape[0],
            'removed_rows': self.original_shape[0] - self.df.shape[0]
        }
        
        return cleaning_results
    
    def create_features(self):
        """
        Create additional features for analysis
        """
        # Total orders and value
        self.df['total_orders'] = (
            self.df['order_num_total_ever_online'] + 
            self.df['order_num_total_ever_offline']
        )
        
        self.df['total_value'] = (
            self.df['customer_value_total_ever_offline'] + 
            self.df['customer_value_total_ever_online']
        )
        
        # Average order value
        self.df['avg_order_value'] = self.df['total_value'] / self.df['total_orders']
        
        # Customer lifetime in days
        self.df['customer_lifetime_days'] = (
            self.df['last_order_date'] - self.df['first_order_date']
        ).dt.days
        
        # Days since last order
        self.df['days_since_last_order'] = (
            datetime.now() - self.df['last_order_date']
        ).dt.days
        
        # Channel preference
        self.df['channel_preference'] = np.where(
            self.df['order_num_total_ever_online'] > self.df['order_num_total_ever_offline'],
            'Online',
            np.where(
                self.df['order_num_total_ever_online'] < self.df['order_num_total_ever_offline'],
                'Offline',
                'Multi-channel'
            )
        )
        
        # Channel migration
        self.df['channel_migration'] = np.where(
            self.df['first_order_channel'] == self.df['last_order_channel'],
            'No Migration',
            f"{self.df['first_order_channel']} to {self.df['last_order_channel']}"
        )
        
        return self.df.columns.tolist()

    def calculate_rfm(self):
        """
        Calculate RFM (Recency, Frequency, Monetary) scores
        """
        # Calculate R, F, M scores
        self.df['R'] = pd.qcut(
            self.df['days_since_last_order'],
            q=5,
            labels=[5, 4, 3, 2, 1]
        )
        
        self.df['F'] = pd.qcut(
            self.df['total_orders'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5]
        )
        
        self.df['M'] = pd.qcut(
            self.df['total_value'].rank(method='first'),
            q=5,
            labels=[1, 2, 3, 4, 5]
        )
        
        # Calculate RFM Score
        self.df['RFM_Score'] = (
            self.df['R'].astype(str) +
            self.df['F'].astype(str) +
            self.df['M'].astype(str)
        )
        
        # Segment customers based on RFM Score
        def segment_customers(row):
            if row['R'] >= 4 and row['F'] >= 4 and row['M'] >= 4:
                return 'Champions'
            elif row['R'] >= 3 and row['F'] >= 3 and row['M'] >= 3:
                return 'Loyal Customers'
            elif row['R'] >= 3 and row['F'] >= 1 and row['M'] >= 2:
                return 'Active Customers'
            elif row['R'] >= 2 and row['F'] >= 2 and row['M'] >= 2:
                return 'At Risk'
            else:
                return 'Lost Customers'
                
        self.df['Customer_Segment'] = self.df.apply(segment_customers, axis=1)
        
        return self.df[['master_id', 'R', 'F', 'M', 'RFM_Score', 'Customer_Segment']]

    def analyze_customer_behavior(self):
        """
        Perform comprehensive customer behavior analysis
        """
        analysis_results = {}
        
        # Channel preferences analysis
        channel_analysis = self.df['channel_preference'].value_counts().to_dict()
        analysis_results['channel_preferences'] = channel_analysis
        
        # Customer segments analysis
        segment_analysis = self.df['Customer_Segment'].value_counts().to_dict()
        analysis_results['customer_segments'] = segment_analysis
        
        # Category interests analysis
        if 'interested_in_categories_12' in self.df.columns:
            category_analysis = (
                self.df.groupby('interested_in_categories_12')
                .agg({
                    'master_id': 'count',
                    'total_value': 'mean',
                    'total_orders': 'mean'
                })
                .round(2)
                .to_dict()
            )
            analysis_results['category_analysis'] = category_analysis
        
        # Customer value analysis
        value_analysis = {
            'avg_customer_value': self.df['total_value'].mean(),
            'median_customer_value': self.df['total_value'].median(),
            'avg_order_value': self.df['avg_order_value'].mean()
        }
        analysis_results['value_analysis'] = value_analysis
        
        return analysis_results

    def generate_visualizations(self):
        """
        Generate key visualizations for the analysis
        """
        # Set up the plotting style
        plt.style.use('seaborn')
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Channel Preference Distribution
        plt.subplot(2, 2, 1)
        sns.countplot(data=self.df, x='channel_preference')
        plt.title('Distribution of Channel Preferences')
        plt.xticks(rotation=45)
        
        # 2. Customer Segment Distribution
        plt.subplot(2, 2, 2)
        sns.countplot(data=self.df, x='Customer_Segment')
        plt.title('Distribution of Customer Segments')
        plt.xticks(rotation=45)
        
        # 3. Average Order Value by Channel
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.df, x='channel_preference', y='avg_order_value')
        plt.title('Average Order Value by Channel')
        plt.xticks(rotation=45)
        
        # 4. Customer Lifetime Value Distribution
        plt.subplot(2, 2, 4)
        sns.histplot(data=self.df, x='total_value', bins=50)
        plt.title('Distribution of Customer Lifetime Value')
        
        plt.tight_layout()
        return fig

def main():
    # Load the data
    try:
        df = pd.read_csv('your_data.csv')  # Replace with your data source
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize the processor
    processor = CRMDataProcessor(df)
    
    # Execute ETL pipeline
    print("Starting ETL process...")
    
    # 1. Clean the data
    cleaning_results = processor.clean_data()
    print("\nCleaning Results:")
    print(cleaning_results)
    
    # 2. Create features
    new_features = processor.create_features()
    print("\nNew Features Created:")
    print(new_features)
    
    # 3. Calculate RFM
    rfm_results = processor.calculate_rfm()
    print("\nRFM Analysis Completed")
    
    # 4. Analyze customer behavior
    behavior_analysis = processor.analyze_customer_behavior()
    print("\nCustomer Behavior Analysis:")
    print(behavior_analysis)
    
    # 5. Generate visualizations
    print("\nGenerating visualizations...")
    visualizations = processor.generate_visualizations()
    
    print("\nETL and analysis pipeline completed successfully!")
    
    return processor.df

if __name__ == "__main__":
    main()
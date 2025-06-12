# config/features.py

features = {
    "target_market": ["Order Item Cardprod Id", "Order Item Product Price", "Order Region", "Product Price"],
    "product_price": ["Category Name", "Department Name", "Sales", "Order Region", "Product Image"],
    "sales_per_customer": ["Days for shipping (real)", "Department Id", "Order City", "Order Item Id", 
                           "Order Item Product Price", "Order Item Quantity", "Sales", "Order Item Total", 
                           "Order Region", "Product Card Id", "Product Category Id"],
    "shipping_days": ["Days for shipment (scheduled)", "Late_delivery_risk", "Order Item Cardprod Id"]
}

model_files = {
    "target_market": "models/Market_model.joblib",
    "product_price": "models/product_price_model.joblib",
    "sales_per_customer": "models/sales_per_customer_model.joblib",
    "shipping_days": "models/shipping_model.joblib"
}

prediction_columns = {
    "target_market": "Target Market Prediction",
    "product_price": "Predicted Product Price",
    "sales_per_customer": "Predicted Sales Per Customer",
    "shipping_days": "Predicted Shipping Days"
}

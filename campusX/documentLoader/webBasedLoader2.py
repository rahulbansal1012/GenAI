from langchain_community.document_loaders import WebBaseLoader 

url =  "https://www.flipkart.com/triggr-kraken-x1-battery-display-40ms-latency-quad-mic-enc-40-hr-v5-3-bluetooth/p/itm9c05f498c3463?pid=ACCGQ2SND4UPDYAF&lid=LSTACCGQ2SND4UPDYAF4AJWHE&marketplace=FLIPKART&store=0pm%2Ffcn&srno=b_1_1&otracker=browse&fm=organic&iid=en_OIdggdmeeo6SjLqI2367o6iteu6-1LB-1F2FZaogvDlONPWzPKpkylhedjHZtt6azsKcLJROUxrCmbXvh8z8uvUFjCTyOHoHZs-Z5_PS_w0%3D&ppt=browse&ppn=browse&ssid=1gqeqh5yz40000001753984852340"
loader =  WebBaseLoader(url)
documents = loader.load()
print("length::",len(documents))
print("type::" ,type(documents))
print("content::" ,documents[0].page_content  )
library(raster)
library(sp)
library(rgdal)
library(rgeos)

#Below are examples of yield maps formats obtained from farmers

#1. Point shapefile
"C:/Flurosat/Modelling/James_town_data/2017 Yield Data/Avonmore Fogdens.shp"


#2. Excel worksheet yield data
"C:/Flurosat/Modelling/HMAg/CSV_yield_filed/Terraidi_field1_Exported CSV Points Cotton Yield.CSV"

#3. 4-band yield raster with grey scale values

"C:/Flurosat/Modelling/HMAg/4_band_raster/Terriadi Field 1 Cotton Yield UTM.tif"


#4. One band yield raster (so far this has not been provided but I converted a point yield shpefile to raster incase we would have this in the future)
"C:/Flurosat/Modelling/James_town_data/Yield_rasters/Avonmore Fogdens.tif"


###Functions

#1. Function to check and return the extension of a yiled data(tif,csv,shp) of the dataset

check_extension<-function(yield_file){
  #obtain input data extention 
  file_extension<-unlist(strsplit(tail(unlist((strsplit(yield_file,"/"))),n=1),"\\."))[2]
  return(file_extension)
}

#Function to read a yield shape file stored in the computer

shp_file_read<-function(yield_file){
  vector_of_names<-unlist((strsplit(yield_file,"/")))
  yield_file_name<-tail(vector_of_names,n=1)
  yield_file_name_no_extension<-unlist(strsplit(yield_file_name,"\\."))[1]
  file_path_without_file_name<-paste(vector_of_names [! vector_of_names %in% yield_file_name],collapse="/")
  
  read_shp_file<-readOGR(file_path_without_file_name,yield_file_name_no_extension)
  yield_data<-subset(read_shp_file,select=Dry_Yield)##we assume that the yield file has got a column named "Dry_Yield", as in the case of data we obtained from South Australia yiled data
  return(yield_data)
}

#Function to read in a yield raster file
raster_file_read<-function(yield_file){
  yield_data<-raster(yield_file)
  return(yield_data)
}











#2. Function to check the format(e.g. point shapefile, raster and csv). of the yield file and use the appropriate function to read the data in. It further checks if the data is meaningful
 check_file_format_and_read_data<-function(yield_file){
   file_extension<-check_extension(yield_file)
   if((match(tolower(file_extension),"tif"))!is.na){
     yield_data<-raster(yield_file)
   }
   else if((match(tolower(file_extension),"shp"))!is.na){
     #we assume that the yield file has got a column named "Dry_Yield"
   }
   
   else if((match(tolower(file_extension),"csv"))!is.na){
     yield_data<-shp_file_read(yield_file)
   }
   
 }
 
 
 
 
 
 
 
 
#####Correlation
 
rasters_list<-list.files("C:/Flurosat/Modelling/HMAg/raster/Terraidi_field_1",full.names =T)
rasters_list<-rasters_list[-18]
#sample_rast<-raster(rasters_list[1])

csv_file<-read.csv("C:/Flurosat/Modelling/HMAg/CSV_yield_filed/Terraidi_field1_Exported CSV Points Cotton Yield.CSV",header=T)
csv_file<-csv_file[complete.cases(csv_file),]
coordinates(csv_file)=~UTM_Easting+UTM_Northing
crs(csv_file)<- "+proj=utm +zone=55 +south +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0"
#shp_utm <- spTransform(csv_file, crs(sample_rast))

#randomly select some yield point
set.seed(1)
random_points<-sample(1:nrow(shp_utm),floor(nrow(shp_utm)/5),replace=FALSE)
subset_yield_points<-shp_utm[random_points,]


###Do some extraction
Cotton_yield<-subset_yield_points$Cotton.Yield
n=1
for(raster_file in rasters_list){
  raster_ndre<-raster(raster_file)
  ndre_raster<-extract(raster_ndre,subset_yield_points,SP=F,df=T)
  if(n==1){
    ndre_yield_table<-cbind(Cotton_yield,ndre_raster)
  }
  
  else{
    ndre_yield_table<-cbind(ndre_yield_table,ndre_raster)
  }
  n<-n+1
}



ndre_yield_table<-ndre_yield_table[,-which("ID"==names(ndre_yield_table))]
ndre_data<-ndre_yield_table[complete.cases(ndre_yield_table),]
cor_data<-cor(ndre_data)
yield_cor<-unname(cor_data[1,2:24])
image_dates<-names(cor_data[1,2:24])

all_raster_date<-c()
for(image_date in rasters_list){
  year<-substr(image_date,1,4)
  mm<-substr(image_date,5,6)
  dd<-substr(image_date,7,8)
  date_value<-paste(c(dd,mm,year),collapse="-")
  all_raster_date<-c(all_raster_date,date_value)
}

par(mar=c(6,4,1,1))
barplot(yield_cor,names.arg =all_raster_date,main="Yield vs NDRE Correlation",ylab="correlation coefficient",xaxt ="n")
#axis(1:30,y=0,labels=all_raster_date,srt=45)

axis(1,at=as.vector(b),labels=all_raster_date,las=2)

###################################################################################
#####Function to convert yield point shapefile to raster

###Rasterise yield and sample both yield and vI raster 
#######################################################################################
#Function
###Function
##Function



yield_points_to_raster<-function(yield_shp_data,image_raster_file_name){
  if(is.character(yield_shp_data)){
    vector_of_names<-unlist((strsplit(yield_shp_data,"/")))
    yield_file_name<-tail(vector_of_names,n=1)
    yield_file_name_no_extension<-unlist(strsplit(yield_file_name,"\\."))[1]
    file_path_without_file_name<-paste(vector_of_names [! vector_of_names %in% yield_file_name],collapse="/")
    read_shp_file<-readOGR(file_path_without_file_name,yield_file_name_no_extension)
  }
  
  else{
    read_shp_file<-yield_shp_data
  }
  
  if(is.character(image_raster_file_name)){
    one_sample_VI_ras<-raster(image_raster_file_name)
  }
  else{
    one_sample_VI_ras<-image_raster_file_name
  }
  
  one_sample_VI_ras<-aggregate(one_sample_VI_ras,resol)
  #Please remember to generalize the yield column name retrieval on the shapefile. Easier to identify the column name in the csv file from Terraidi as it's got one column besides coordinates 
  Yield<-rasterize(read_shp_file, one_sample_VI_ras, read_shp_file$Cotton.Yield, fun = mean)
  return(Yield)
}





##Kml to shpfile
kml_file<-"C:/Users/user/Downloads/kmls_20180509002047/G & C Houston - Terriadi - 1.kml"
farm_boundry<- readOGR(kml_file,"OGRGeoJSON")

######lay_bound<-readOGR(dsn=lay_bound_folder,layer="Dooling_farming")
random_points<-spsample(farm_boundry,n=100,type="regular")

plot(farm_boundry)
plot(random_points,add=T,type="o")


#analysis
yield_ras<-yield_points_to_raster(csv_file,rasters_list[1])

if(crs(random_points)@projargs!=crs(yield_ras)@projargs){
  random_points<-spTransform(random_points,crs(yield_ras))
}

yield<-extract(yield_ras,random_points)



n=1

for(raster_file in rasters_list){
  raster_ndre<-raster(raster_file)
  ndre_raster<-extract(raster_ndre,random_points,SP=F,df=T)
  if(n==1){
    ndre_yield_table<-cbind(yield,ndre_raster)
  }
  
  else{
    ndre_yield_table<-cbind(ndre_yield_table,ndre_raster)
  }
  n<-n+1
}

ndre_yield_table<-ndre_yield_table[,-which("ID"==names(ndre_yield_table))]
ndre_data<-ndre_yield_table[complete.cases(ndre_yield_table),]
cor_data<-cor(ndre_data)
yield_cor<-unname(cor_data[1,2:24])
image_dates<-names(cor_data[1,2:24])

all_raster_date<-c()
for(image_date in image_dates){
  year<-substr(image_date,2,5)
  mm<-substr(image_date,6,7)
  dd<-substr(image_date,8,9)
  date_value<-paste(c(dd,mm,year),collapse="-")
  all_raster_date<-c(all_raster_date,date_value)
}

par(mar=c(6,4,1,1))
b<-barplot(yield_cor,names.arg =all_raster_date,ylim=c(min(yield_cor),0.4),main="Yield vs NDRE Correlation",ylab="correlation coefficient",xaxt ="n")
#axis(1:30,y=0,labels=all_raster_date,srt=45)

axis(1,at=as.vector(b),labels=all_raster_date,las=2)



#####analysis per zoning average
make_management_zone_data_frame_from_yield_and_veg_raster2<-function(yield_shp_file,zoning_shp_file,rasts){
  #mgt zone shapafile
  if(crs(yield_shp_file)@projargs!=crs(zoning_shp_file)@projargs){
    zoning_shp_file<-spTransform(zoning_shp_file,crs(yield_shp_file))
  }
  
  if(crs(yield_shp_file)@projargs!=crs(rasts)@projargs){
    rasts<-projectRaster(rasts,res=20,crs=crs(yield_shp_file))
  }
  zoning_shp_file<-spTransform(zoning_shp_file,crs(rasts))
  #shpfile yield
  mgt_yield_shape<-intersect(yield_shp_file,zoning_shp_file)
  
  combine_data<-as.data.frame(extract(rasts,mgt_yield_shape,df=T,sp=T))
  
  combine_data2<-combine_data[,3:(ncol(combine_data))]
  
  ras_name<-names(combine_data2)[ncol(combine_data2)]
  year<-substr(ras_name,2,5)
  mm<-substr(ras_name,6,7)
  dd<-substr(ras_name,8,9)
  date_value<-paste(c(dd,mm,year),collapse="-")
  print(date_value)
  #heading of data
  colnames(combine_data2)<-c("Yield","Zones",date_value)
  
  
  combine_data3<-combine_data2[complete.cases(combine_data2),]
  
  as.factor(combine_data3$Zones)
  
  combine_data4<-aggregate(.~Zones,data=combine_data3,FUN="mean")
  cor_coff<-cor(combine_data4[,2],combine_data4[,3])
  return(c(cor_coff,date_value))
}


par(mar=c(6,4,4,1))
b<-barplot(h$cor_vals,names.arg =h$date_vals,ylim=c(0,1),main="Yield vs NDRE Image Correlation",ylab="correlation coefficient",xaxt ="n")

axis(1,at=as.vector(b),labels=h$date_vals,xlab="image date acquisition",las=2)







zone_shp_folder_list<-list.files("C:/Flurosat/Modelling/HMAg/zone_shape_Ndre_Terraidi",full.names = T)
n<-18
cor_vals<-c()
date_vals<-c()
for(zon_shp in zone_shp_folder_list[18:24]){
  zone_mgt_name<-readOGR(zon_shp,"classes")
  rasts<-raster(rasters_list[n])
  result<-make_management_zone_data_frame_from_yield_and_veg_raster2(csv_file,zone_mgt_name,rasts)
  cor_vals<-c(cor_vals,result[1])
  date_vals<-c(date_vals,result[2])
  n<-n+1
  print(n)
}

length(cor_vals)
length(date_vals)

yield_ndre_by_zone_corr<-as.data.frame(cbind(cor_vals,date_vals))
yield_ndre_by_zone_corr$cor_vals<-as.numeric(as.character(yield_ndre_by_zone_corr$cor_vals))
write.csv(yield_ndre_by_zone_corr,"C:/Flurosat/Modelling/HMAg/yield_ndre_by_zone_correlation_2.csv")
h<-read.csv("C:/Flurosat/Modelling/HMAg/yield_ndre_by_zone_correlation_final.csv",header=T)


par(mar=c(6,4,4,1))
b<-barplot(yield_ndre_by_zone_corr$cor_vals,names.arg =yield_ndre_by_zone_corr$date_vals,ylim=c(0,1),main="Yield vs NDRE Image Correlation",ylab="correlation coefficient",xaxt ="n")

axis(1,at=as.vector(b),labels=yield_ndre_by_zone_corr$date_vals,xlab="image date acquisition",las=2)




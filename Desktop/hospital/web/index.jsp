<%-- 
    Document   : index
    Created on : 9 Jul, 2018, 2:04:47 PM
    Author     : trainee
--%>

<%@page contentType="text/html" pageEncoding="UTF-8"%>
<%@ page import = "java.io.*,java.util.*,java.sql.*"%>
<%@ page import = "javax.servlet.http.*,javax.servlet.*" %>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="style.css" />
<style>
    body html{
        margin: 0;
        padding: 0;
    }
   
    .backgroundimg {
        
        background-image:url('https://d2v9y0dukr6mq2.cloudfront.net/video/thumbnail/rd7hGetRgj1tekoj0/videoblocks-medical-staff-at-the-hospitals-corridor-blurred-background-timelapse_ba615vcce_thumbnail-full01.png');
            
        background-size: cover;
        
        background-position: center center;    
        
        background-repeat: no-repeat;
        
        height: 100vh;
        
        width: 100%;
    }
.nav-tabs{
  opacity: 1;
    filter: alpha(opacity=50);
    font-weight: bold;
    padding-top: 15px;
}
.nav-tabs > li > a{
  border: medium none;
}
.nav-tabs > li > a:hover{
  background-color: #303136 !important ;
    border: medium none;
    border-radius: 0;
    color:#ffffff ;
}
.color-me{
    color:#ffffff;
}
.h1, h2{
    text-shadow: 2px 2px 4px #000000;
    font-size: 3.175em!important;
    color:#ffffff ;
}
#my-div{
 
  margin: 0 auto;
  width: 80%;
  height: 50%;
}
</style>

</head>
    
    <body> 
        <div id="my-div">
        <div class="row backgroundimg">
            <div class="container">        
    <img style="float:left" src="http://ants2017.ieee-comsoc-ants.org/files/2017/07/cdotlogo-2-191x300.jpg" width="100" height="150">
                <h1 class="h1">&nbsp APOLLO HOSPITAL</h1>
                <br/>
                <br/>
                <br/>
                <hr>
                <br/>
                <br/>
                <h2 >Login </h2>
               
                <form  action="verify.jsp" method="get"  >
                    <a class="color-me" >Enter UserName:&nbsp&nbsp</a><input type="text" name="uname"/> 
                    <br/>
                    <a class="color-me" >Enter password:&nbsp&nbsp&nbsp&nbsp</a><input type="password" name="pass"/>
                    <br/><input type="submit" class="btn btn-success"/>                    
                </form>
                
  <%-- <button class="tablinks"  style="text-decoration:none" onclick="library(event, 'total')" >Available books</button>
  <button class="tablinks"  style="text-decoration:none" onclick="library(event, 'issued')">Issued books</button>
  <button class="tablinks"  style="text-decoration:none" onclick="library(event, 'current')">Currently available books</button>
 --%>
        </div>
        </div>
        </div>
</body>

</html>
     
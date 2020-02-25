<%@page import="java.sql.DriverManager"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.Statement"%>
<%@page import="java.sql.Connection"%>

<html>
    <head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="style.css" />
  <meta charset="utf-8">
<style>
body html{
        margin: 0;
        padding: 0;
    }
   
    .backgroundimg {
        
        background-image:url('https://d2v9y0dukr6mq2.cloudfront.net/video/thumbnail/B1eYNZGrripnnry82/on-the-table-at-library-blurred-background_hrbq_xtkx_thumbnail-full01.png');
            
        background-size: cover;
        
        background-position: center center;    
        
        background-repeat:repeat;
                
        height: 100vh;
        
        width: 100%;
       
  -webkit-background-size: cover;
  -moz-background-size: cover;
  -o-background-size: cover;
  background-size: cover;
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
.color-me1{
    color:  #ff0000  ;
    font-weight: bold;
    text-shadow: 2px 2px 4px #000000;
}
.h1, h1 {
    font-size: 3.175em!important;
    color:#ffffff ;
    text-shadow: 2px 2px 4px #000000;
}
div.ex1 {
    opacity: 1;
    filter: alpha(opacity=50);
    font-weight: bold;
    height: 400px;
    width: 875px;
    overflow-y: scroll;
    position:center ;
    overflow: auto;
   
}
.center {
    margin: auto;
    width: 60%;
    padding: 10px;
    
}
table {
    color: white;
    font-size: 140%;
}
input[type=submit]
{
width:10%;
color: aliceblue;
font-size:15px;
cursor:pointer;
}
#submit:hover
{
background-color: black;
}
#my-div{
  
  margin: 0 auto;
  
  width: 80%;
  height: 50%;
}
</style>

</head>
<div id="my-div">
<div class="row backgroundimg">
            <div class="container">
                 <img style="float:left" src="http://ants2017.ieee-comsoc-ants.org/files/2017/07/cdotlogo-2-191x300.jpg" width="100" height="150">
                <h1 class="h1">&nbsp APOLLO HOSPITAL</h1>
   <ul class="nav nav-tabs">
       <li><a href="doctors.jsp" class="color-me" >DOCTORS</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</li>
     <li><a href="nurse.jsp" class="color-me">NURSE</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</li>
     <li><a href="patient.jsp" class="color-me">PATIENTS</a></li>
     <li><a href="addpat.jsp" class="color-me">+ADD PATIENT</a></li>
       </ul> 
<%
//String id = request.getParameter("userid");
String driver = "com.mysql.jdbc.Driver";
String connectionUrl = "jdbc:mysql://localhost/";
String database = "hospital";
String userid = "root";
String password = "mysql@root";
try {
Class.forName(driver);
} catch (ClassNotFoundException e) {
e.printStackTrace();
}
Connection connection = null;
Statement statement = null;
ResultSet resultSet = null;
%>
<!DOCTYPE html>
<center><h2 class="color-me1" >PATIENTS</h2></center> 
<%--<center><form action="nurse.jsp" method="POST">
    <select name="value" >      
    <option value="cardioligist">Cardiologist</option>    
    <option value="psychiatrist">Psychiatrist </option>     
    <option value="dentist">Dentist</option>
    <option value="%">All</option>
    
    </select>   
        <input type="submit" class="btn btn-primary" value="submit"/>
   
    </form></center> --%>
<div class="ex1 center" >
    <center><table border="1">
<tr >
<b><td  style="font-weight:bold" bgcolor="#000000">PATIENT_ID</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">NAME</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">DOCTOR</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">PHONE NO</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">AGE</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">GENDER</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">DISEASE</td></b>
</tr>
<%
//String val=request.getParameter("value");
try{
connection = DriverManager.getConnection(connectionUrl+database, userid, password);
statement=connection.createStatement();
String sql ="select p.patient_id,p.name,d.name,p.phone_no,p.age,p.gender,p.disease from doctors d,patient p where p.doctor_id=d.doctor_id ;";
resultSet = statement.executeQuery(sql);
while(resultSet.next()){
%>
<tr>
<td><%=resultSet.getInt("p.patient_id") %></td>
<td><%=resultSet.getString("p.name") %></td>
<td><%=resultSet.getString("d.name") %></td>
<td><%=resultSet.getString("p.phone_no") %></td>
<td><%=resultSet.getInt("p.age") %></td>
<td><%=resultSet.getString("p.gender") %></td>
<td><%=resultSet.getString("p.disease") %></td>
</tr>
<%
}
connection.close();
} catch (Exception e) {
e.printStackTrace();
}
%>
</table></center> 
</div>
            </div>
</div>
</div>
</html>
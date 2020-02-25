<%@page import="javax.swing.JOptionPane"%>
<%@page import="java.sql.DriverManager"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.Statement"%>
<%@page import="java.sql.Connection"%>
<%@page import="java.util.*"%>

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
.b1{
    
    position:center ;
}
div.ex1 {
    opacity: 1;
    filter: alpha(opacity=50);
    font-weight: bold;
    height: 400px;
    width: 950px;
    overflow-y: scroll;
    position:center ;
    overflow: auto;
}
.center {
    margin: auto;
    width: 50%;
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
.modal-header {
    padding:9px 15px;
    border-bottom:1px solid #eee;
    background-color: #0480be;
    -webkit-border-top-left-radius: 5px;
    -webkit-border-top-right-radius: 5px;
    -moz-border-radius-topleft: 5px;
    -moz-border-radius-topright: 5px;
     border-top-left-radius: 5px;
     border-top-right-radius: 5px;
 }
</style>

</head>  
<body>
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
  <center><h2 class="color-me1">+ADD PATIENT</h2></center> 
                <!-- Trigger the modal with a button -->
                <center><div class="btn-group">
                <button type="submit" class="btn btn-primary" data-toggle="modal" data-target="#myModal" >Add Patient</button>
                <button type="submit" class="btn btn-danger" data-toggle="modal" data-target="#myModal2" >Delete Patient</button>
                </div></center>
  <!-- Modal -->
  <div class="modal fade" id="myModal" role="dialog">
    <div class="modal-dialog">
    
      <!-- Modal content-->
      <div class="modal-content">
          
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal">&times;</button>
          <h4 class="modal-title" style="color:white;" >ADD PATIENT</h4>
        </div>
          
        <div class="modal-body">
       <form action="addpat.jsp">
    <div class="form-group">
     <%-- <label>Patient ID: </label>
      <input type="number" class="form-control" placeholder="Enter employee ID" name="emp_id">
    </div>
    
 --%>
      <label>Category: </label>
     <div class="input-group">
            <select  style="width:175px;" name="value">      
    <option value="1">Cardiologist</option>    
    <option value="2">Psychiatrist </option>     
    <option value="3">Dentist</option>
    </select>
          
   </div>
           
    <div class="form-group">
      <label> Patient Name:  </label>
      <input type="Text" class="form-control" placeholder="Enter patient name" name="name">
    </div>
           
    <div class="form-group">
      <label> Phone number:  </label>
      <input type="Text" class="form-control" placeholder="Enter phone number" name="phone">
    </div>
           
    <div class="form-group">
      <label> Age:  </label>
      <input rows="4" class="form-control" placeholder="Enter age" cols="50" name="age" >
    </div>
       <div class="input-group">
            <select  style="width:175px;" name="value1">      
    <option value="male">Male</option>    
    <option value="female">Female </option>     
    </select>
          
   </div>
      <div class="form-group">
      <label> Disease:  </label>
      <textarea type="Text" class="form-control" placeholder="description of the problem" name="disease"></textarea>
    </div>
           
      <button type="submit" class="btn btn-success" name="addrec">Submit</button>
  </form> 
        </div>
         
      </div>
      
    </div>
  </div>
  </div>
   <div class="modal fade" id="myModal2" role="dialog">
                        <div class="modal-dialog">
    
                        <!-- Modal content-->
                        <div class="modal-content">
          
                            <div class="modal-header">
                            <button type="button" class="close" data-dismiss="modal">&times;</button>
                            <h4 class="modal-title" style="color:white;" >DELETE</h4>
                            </div>
          
                        <div class="modal-body">
                            <form action="addpat.jsp">
                            <div class="form-group">
                                <label>Patient ID: </label>
                                <input type="number" class="form-control" placeholder="Enter patient ID" name="pat_id1">
                            </div>
           <%--
                           <div class="form-group">
                                <label> Employee Name:  <small>(optional)</small></label>
                                <input type="Text" class="form-control" placeholder="Enter employee name" name="name1">
                            </div>
           --%>
                                <button type="submit" class="btn btn-danger" name="delrec">Delete</button>
                            </form> 
                         </div>     
                         </div>
                      </div>
                  
   </div>
           
<%--  
   <div class="modal fade" id="myModal1" role="dialog">
    <div class="modal-dialog">
    
      <!-- Modal content-->
      <div class="modal-content">
          
        <div class="modal-header">
          <button type="button" class="close" data-dismiss="modal">&times;</button>
          <h4 class="modal-title" style="color:white;" >DELETE PATIENT</h4>
        </div>
          
        <div class="modal-body">
       <form action="addpat.jsp">
    <div class="form-group">
      <label>Patient ID: </label>
      <input type="number" class="form-control" placeholder="Enter Patient ID" name="pat_id1">
    </div>
          
           
           <button type="submit" class="btn btn-danger" name="delrec">Delete</button>
  </form> 
        </div>
         
      </div>
      
    </div>
  </div>
  
--%>
                
<%
   /* int pick=0;
    Random rand = new Random(3);
     for (int j = 3; j<100; ++j)
     {
       pick = rand.nextInt(100);
        }
*/
if (request.getParameter("addrec") != null) {
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
ResultSet nm = null;
ResultSet tt = null;
ResultSet tit = null;
String empn=null;
String tt1=null;
String tit1=null;
try{
    int empid= 1000;
    if(empid!=0){
    String cat = request.getParameter("value");
    String name= request.getParameter("name");
    String phone= request.getParameter("phone");
    String age= request.getParameter("age");
    String value1 = request.getParameter("value1");
    String disease= request.getParameter("disease");
connection = DriverManager.getConnection(connectionUrl+database, userid, password);
statement=connection.createStatement();
//String sql1 ="select title from booksdb where title like '"+title+"';";
//tt= statement.executeQuery(sql1);
//while(tt.next()){
  //tt1 = tt.getString("title");
//}
//String sql4 ="select title from recommendation where title like '"+title+"';";
//tit= statement.executeQuery(sql4);
//while(tit.next()){
 // tit1 = tit.getString("title");
//}


//if(tt1!=null){
//JOptionPane.showMessageDialog(null, "This book already exists");

//}
//if(tit1!=null)

//{
  //  JOptionPane.showMessageDialog(null, "This book has already been recommended");
//}
//else if(tt1==null && tit1==null){
//String sql2 ="select name from emp where empid="+empid+";";
//nm = statement.executeQuery(sql2);
//while(nm.next()){
 // empn = nm.getString("name");
//}
String sql ="insert into patient(doctor_id,name,phone_no,age,gender,disease)values('"+cat+"','"+name+"','"+phone+"',"+age+",'"+value1+"','"+disease+"');";
statement.executeUpdate(sql);
}
} 
catch (Exception e) {
e.printStackTrace();
}
}

if (request.getParameter("delrec") != null) {
String driver9= "com.mysql.jdbc.Driver";
String connectionUrl9= "jdbc:mysql://localhost/";
String database9= "hospital";
String userid9= "root";
String password9 = "mysql@root";
Connection connection9 = null;
Statement statement9 = null;
ResultSet resultSet9 = null;
try{
     //int empid1= pick;
    int pat_id1= Integer.parseInt(request.getParameter("pat_id1"));
connection9 = DriverManager.getConnection(connectionUrl9+database9, userid9, password9);
statement9=connection9.createStatement();
String sql9 ="delete from patient where patient_id="+pat_id1+";";
statement9.executeUpdate(sql9);
}
catch (Exception e) {
e.printStackTrace();
}
}

String driver7= "com.mysql.jdbc.Driver";
String connectionUrl7= "jdbc:mysql://localhost/";
String database7= "hospital";
String userid7= "root";
String password7 = "mysql@root";
Connection connection7 = null;
Statement statement7 = null;
ResultSet resultSet7 = null;
try{
connection7 = DriverManager.getConnection(connectionUrl7+database7, userid7, password7);
statement7=connection7.createStatement();
String sql3 ="select * from patient;";
resultSet7 = statement7.executeQuery(sql3);
 int flag=1;
while(resultSet7.next()){
     if(flag==1)
                                        {
                                            flag=2;
         
%>


<div class="ex1 center" >
    <center><table border="1">
<tr >
<b><td style="font-weight:bold" bgcolor="#000000">PATIENT ID</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">DOCTOR ID</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">NAME</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">PHONE NUMBER</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">AGE</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">GENDER</td></b>
<b><td style="font-weight:bold" bgcolor="#000000">DISEASE</td></b>
</tr>
 <%}%>
<tr>
<td><%=resultSet7.getInt("patient_id") %></td>
<td><%=resultSet7.getString("doctor_id") %></td>
<td><%=resultSet7.getString("name") %></td>
<td><%=resultSet7.getString("phone_no") %></td>
<td><%=resultSet7.getString("age") %></td>
<td><%=resultSet7.getString("gender") %></td>
<td><%=resultSet7.getString("disease") %></td>
</tr>
<%
}
//connection.close();
} catch (Exception e) {
e.printStackTrace();
}
%>
        </table></center>
</div>
       
            </div>
              </div>
</div>
</body>
</html>

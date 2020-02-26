<%-- 
    Document   : verify.jsp
    Created on : 24 Jul, 2018, 2:54:25 PM
    Author     : trainee
--%>
<%@page import="javax.swing.JOptionPane"%>
<%@page contentType="text/html" pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>JSP Page</title>
    </head>
    <body>
       <%! String uname, pass; %>
       <%
           uname=request.getParameter("uname");
           pass=request.getParameter("pass");
           if(uname.equals("")&&pass.equals(""))
           {
                %>
                <jsp:forward page="home.jsp"/>
                <%}else
           {%>
           <jsp:include page="index.jsp"/>
           <%
           JOptionPane.showMessageDialog(null, "incorrect username or password");
           }%>
    </body>
</html>

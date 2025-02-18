import * as React from "react";
import Box from "@mui/material/Box";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import HomeIcon from "@mui/icons-material/Home";
import AssessmentIcon from "@mui/icons-material/Assessment";
import DeleteIcon from "@mui/icons-material/Delete";
import { Link, useNavigate } from "react-router-dom";
import { Menu } from "lucide-react";
import { useData } from "../context/DataContext";

export default function Sidebar() {
    const navigate = useNavigate();
    const { setData, isSidebarCollapsed, setIsSidebarCollapsed } = useData(); // Get state from context

    const toggleSidebar = () => {
        setIsSidebarCollapsed((prev) => !prev); // Toggle state
    };

    const handleDeleteStorage = () => {
        localStorage.removeItem("chatData");
        setData({ messages: [] });
        alert("Local storage cleared!");
        navigate("/home");
    };

    return (
        <Box
            sx={{
                width: isSidebarCollapsed ? 80 : 270, 
                bgcolor: "#0A0F24",
                color: "white",
                height: "93vh",
                display: "flex",
                flexDirection: "column",
                paddingTop: 2,
                position: "fixed",
                left: 0,
                top: 0,
                transition: "width 0.3s ease",
                borderRadius: "12px",
                border: "2px solid white",
                boxShadow: "0px 0px 10px rgba(255, 255, 255, 0.1)",     
                marginTop: "20px",
                marginLeft: "20px",
            }}
        >
            <div className="flex items-center justify-between p-4">
                {!isSidebarCollapsed && (
                    <h1 className="text-3xl font-bold tracking-widest text-white">CHATTIBOT</h1>
                )}
                <Menu color="white" size={30} onClick={toggleSidebar} className="cursor-pointer ml-[10px]" />
            </div>
            <Divider sx={{ bgcolor: "#1C233D" }} />
            {!isSidebarCollapsed && (
                <>
                    <List>
                        <ListItem disablePadding>
                            <ListItemButton component={Link} to="/home" sx={{
                                bgcolor: "#131A35",
                                margin: "8px",
                                borderRadius: "8px",
                                color: "white",
                                border: "2px solid white",
                                "&:hover": { bgcolor: "#1C233D", border: "2px solid rgba(255, 255, 255, 0.5)" },
                            }}>
                                <ListItemIcon>
                                    <HomeIcon sx={{ color: "white" }} />
                                </ListItemIcon>
                                <ListItemText primary="Home" />
                            </ListItemButton>
                        </ListItem>
                        <ListItem disablePadding>
                            <ListItemButton component={Link} to="/result" sx={{
                                bgcolor: "#131A35",
                                margin: "8px",
                                borderRadius: "8px",
                                color: "white",
                                border: "2px solid white",
                                "&:hover": { bgcolor: "#1C233D", border: "2px solid rgba(255, 255, 255, 0.5)" },
                            }}>
                                <ListItemIcon>
                                    <AssessmentIcon sx={{ color: "white" }} />
                                </ListItemIcon>
                                <ListItemText primary="Results" />
                            </ListItemButton>
                        </ListItem>
                    </List>
                    <Divider sx={{ bgcolor: "#1C233D", marginTop: "auto" }} />
                    <List>
                        <ListItem disablePadding>
                            <ListItemButton onClick={handleDeleteStorage} sx={{
                                bgcolor: "#131A35",
                                margin: "8px",
                                borderRadius: "8px",
                                color: "red",
                                border: "2px solid white",
                                "&:hover": { bgcolor: "#1C233D", border: "2px solid rgba(255, 255, 255, 0.5)" },
                            }}>
                                <ListItemIcon>
                                    <DeleteIcon sx={{ color: "red" }} />
                                </ListItemIcon>
                                <ListItemText primary="Delete Storage" />
                            </ListItemButton>
                        </ListItem>
                    </List>
                </>
            )}
        </Box>
    );
}

import * as React from "react";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import Button from "@mui/material/Button";
import List from "@mui/material/List";
import Divider from "@mui/material/Divider";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import HomeIcon from "@mui/icons-material/Home";
import AssessmentIcon from "@mui/icons-material/Assessment";
import DeleteIcon from "@mui/icons-material/Delete"; // Import Delete icon
import { Link, useNavigate } from "react-router-dom"; // Import useNavigate
import { Menu } from "lucide-react";
import { useData } from "../context/DataContext";

export default function TemporaryDrawer() {
	const [open, setOpen] = React.useState(false);
	const navigate = useNavigate(); // Initialize useNavigate

	const toggleDrawer = (newOpen) => () => {
		setOpen(newOpen);
	};

	const { setData } = useData();

	const handleDeleteStorage = () => {
		localStorage.removeItem("chatData");
		setData({ messages: [] });
		alert("Local storage cleared!");
		navigate("/home"); // Navigate to /home after deleting storage
	};

	const DrawerList = (
		<Box sx={{ width: 250 }} role="presentation" onClick={toggleDrawer(false)}>
			<div className="text-center p-4">
				<h1 className="text-3xl font-bold tracking-widest">Chattibot</h1>
			</div>
			<Divider />
			<List>
				<ListItem key="Home" disablePadding>
					<ListItemButton component={Link} to="/home">
						<ListItemIcon>
							<HomeIcon sx={{ color: "blue" }} />
						</ListItemIcon>
						<ListItemText primary="Home" />
					</ListItemButton>
				</ListItem>
				{["Result"].map((text) => (
					<ListItem key={text} disablePadding>
						<ListItemButton component={Link} to={`/${text.toLowerCase()}`}>
							<ListItemIcon>
								<AssessmentIcon sx={{ color: "green" }} />
							</ListItemIcon>
							<ListItemText primary={text} />
						</ListItemButton>
					</ListItem>
				))}
			</List>
			<Divider />
			<List>
				<ListItem key="Delete Storage" disablePadding>
					<ListItemButton onClick={handleDeleteStorage} sx={{ color: "red" }}>
						<ListItemIcon>
							<DeleteIcon sx={{ color: "red" }} /> {/* Add bin icon */}
						</ListItemIcon>
						<ListItemText primary="Delete Storage" />
					</ListItemButton>
				</ListItem>
			</List>
		</Box>
	);

	return (
		<React.Fragment>
			<Button onClick={toggleDrawer(true)}>
				<Menu />
			</Button>
			<Drawer anchor="left" open={open} onClose={toggleDrawer(false)}>
				{DrawerList}
			</Drawer>
		</React.Fragment>
	);
}

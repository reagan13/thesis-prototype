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
import InboxIcon from "@mui/icons-material/MoveToInbox";
import MailIcon from "@mui/icons-material/Mail";
import { Link } from "react-router-dom";
import { Menu } from "lucide-react";
import { useData } from "../context/DataContext"; // Import useData

export default function TemporaryDrawer() {
	const [open, setOpen] = React.useState(false);

	const toggleDrawer = (newOpen) => () => {
		setOpen(newOpen);
	};

	const { setData } = useData(); // Get setData from context

	const handleDeleteStorage = () => {
		// Clear local storage
		localStorage.removeItem("chatData"); // Adjust the key if necessary
		// Reset the context data
		setData({ messages: [] }); // Reset messages in context
		alert("Local storage cleared!"); // Optional: Notify the user
		location.reload(); // Reload the page
	};

	const DrawerList = (
		<Box sx={{ width: 250 }} role="presentation" onClick={toggleDrawer(false)}>
			<List>
				<ListItem key="Home" disablePadding>
					<ListItemButton component={Link} to="/home">
						<ListItemIcon>
							<InboxIcon />
						</ListItemIcon>
						<ListItemText primary="Home" />
					</ListItemButton>
				</ListItem>
				{["GPT2", "DistilBERT"].map((text, index) => (
					<ListItem key={text} disablePadding>
						<ListItemButton component={Link} to={`/${text.toLowerCase()}`}>
							<ListItemIcon>
								{index % 2 === 0 ? <InboxIcon /> : <MailIcon />}
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
						<ListItemText primary="Delete Storage" />
					</ListItemButton>
				</ListItem>
			</List>
		</Box>
	);

	return (
		<div>
			<Button onClick={toggleDrawer(true)}>
				<Menu />
			</Button>
			<Drawer open={open} onClose={toggleDrawer(false)}>
				{DrawerList}
			</Drawer>
		</div>
	);
}

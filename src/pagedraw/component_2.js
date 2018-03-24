// Generated by https://pagedraw.io/pages/8950
import React from 'react';
import './component_2.css';
import { BrowserRouter as Router, Route, Link } from "react-router-dom";
import { Redirect } from 'react-router'
// import axios from 'axios';

export default class component2 extends React.Component {
    constructor(props){
        super(props);
        this.state = {
            roomCode: '',
            score: '',
            redirect: false
        }

        this.handleEvent = this.handleEvent.bind(this);
        this.handleClick = this.handleClick.bind(this);
    }

    handleEvent() {
        this.state.roomCode = document.getElementsByClassName('component_2-text_input-6')[0] ? document.getElementsByClassName('component_2-text_input-6')[0].value : ''
        this.state.score = document.getElementsByClassName('component_2-text_input_2')[0] ? document.getElementsByClassName('component_2-text_input_2')[0].value : ''
    }

    handleClick() {
        this.setState({redirect: true});
    }

    render() {
        const { redirect } = this.state;

        if (redirect) {
            return <Redirect to='/bamboozle'/>;
        }

        return (<div className="component_2-component_2">
        <div className="component_2-0">
            <div className="component_2-0-0" /> 
            <div className="component_2-text-7">
                {"Yao's Millionare "}
            </div>
            <div className="component_2-0-2" /> 
        </div>
        <div className="component_2-1">
            <div className="component_2-1-0" /> 
            <div className="component_2-rectangle-8">
                <div className="component_2-1-1-0">
                    <div className="component_2-rectangle_2">
                        <div className="component_2-1-1-0-0-0">
                            <input type="text" placeholder="Room Code" className="component_2-text_input-6" value={this.roomCode} onChange={this.handleEvent}/> 
                        </div>
                    </div>
                </div>
                <div className="component_2-1-1-1">
                    <div className="component_2-rectangle_3">
                        <div className="component_2-1-1-1-0-0">
                            <input type="text" placeholder="Score" className="component_2-text_input_2" value={this.score} onChange={this.handleEvent} /> 
                        </div>
                    </div>
                </div>
                <div className="component_2-1-1-2">
                    <div className="component_2-rectangle_4">
                        <div className="component_2-1-1-2-0-0">
                            <div className="component_2-text_4" onClick={this.handleClick} >Submit</div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="component_2-1-2" /> 
        </div>
        <div className="component_2-2" /> 
        </div>);
    }
}

// export default function Yee2 (props) {
    // return this.component2.render()
// }